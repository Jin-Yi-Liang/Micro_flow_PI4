#ifndef MICROFLOW_GEMM_HPP
#define MICROFLOW_GEMM_HPP

#include "microflow/tensor.hpp"
#include <cstdint>
#include <functional>
#include <vector>

namespace microflow {

/**
 * @brief GEMM配置参数
 *
 * @设计考量:
 * - 不同矩阵大小需要不同的分块策略
 * - 缓存大小影响最优块大小
 * - ARM NEON的向量长度是128位(4个float)
 */
struct GEMMConfig {
    // L1缓存分块 (32KB数据缓存)
    int mc = 32;   // M方向分块
    int nc = 32;   // N方向分块
    int kc = 128;  // K方向分块

    // 寄存器分块 (NEON寄存器数量有限)
    int mr = 4;    // M方向寄存器分块
    int nr = 8;    // N方向寄存器分块

    // 预取距离
    int prefetch_distance = 4;

    // 是否使用FMA指令 (Cortex-A72支持)
    bool use_fma = true;

    // 是否启用循环展开
    bool unroll = true;
};

/**
 * @brief GEMM性能统计
 */
struct GEMMStats {
    uint64_t total_ops = 0;      // 总浮点运算数
    double time_ms = 0.0;        // 执行时间(毫秒)
    double gflops = 0.0;         // GFLOPS
    int cache_misses = 0;        // 缓存miss数(PMU)
};

/**
 * @brief GEMM实现类型
 */
enum class GEMMImpl {
    kNaive,       // 三重循环
    kOpenMP,      // OpenMP并行
    kNEON,        // ARM NEON优化
    kAuto,        // 自动选择最优实现
};

//==========================================================================
// 基础GEMM接口
//==========================================================================

/**
 * @brief 通用矩阵乘法: C = A * B + C
 *
 * @param A 输入矩阵 [M x K]
 * @param B 输入矩阵 [K x N]
 * @param C 输出矩阵 [M x N]
 * @param config GEMM配置
 *
 * @矩阵布局:
 * - 所有矩阵使用Row-Major布局
 * - A: M行K列, stride = K
 * - B: K行N列, stride = N
 * - C: M行N列, stride = N
 *
 * @复杂度: O(M * N * K)
 *
 * @优化策略:
 * 1. 缓存分块 (Cache Blocking)
 * 2. 寄存器分块 (Register Blocking)
 * 3. 循环展开 (Loop Unrolling)
 * 4. NEON SIMD向量化
 * 5. 软件预取 (Software Prefetching)
 */
void gemm(const Tensor& A, const Tensor& B, Tensor& C,
          const GEMMConfig& config = GEMMConfig());

/**
 * @brief Batch-one fully connected kernel using output-major packed weights.
 *
 * @param input Input vector [in_features]
 * @param packed_weights Weights [out_features, in_features]
 * @param bias Optional bias [out_features]
 * @param output Output vector [out_features]
 * @param apply_relu Fuse ReLU into the output store
 */
void gemv_packed(const Tensor& input, const Tensor& packed_weights,
                 const Tensor& bias, Tensor& output, bool apply_relu = false);

/** Dynamic-int8 batch-one fully connected kernel with per-output weight scales. */
void gemv_int8_dynamic(const Tensor& input,
                       const std::vector<int8_t>& packed_weights,
                       const std::vector<float>& weight_scales,
                       const std::vector<int32_t>& weight_sums,
                       const Tensor& bias,
                       Tensor& output,
                       bool apply_relu,
                       std::vector<int8_t>& input_buffer);

/**
 * @brief 批量矩阵乘法
 *
 * @param batch 批大小
 * @param A 批量矩阵 [batch x M x K]
 * @param B 批量矩阵 [batch x K x N]
 * @param C 输出矩阵 [batch x M x N]
 *
 * @优化点:
 * - OpenMP并行化batch维度
 * - 减少小矩阵的启动开销
 */
void batch_gemm(int batch, const Tensor* A, const Tensor* B, Tensor* C,
                const GEMMConfig& config = GEMMConfig());

/**
 * @brief 带转置的GEMM
 *
 * @param A 输入矩阵
 * @param B 输入矩阵
 * @param C 输出矩阵
 * @param transpose_A 是否转置A
 * @param transpose_B 是否转置B
 *
 * @优化点:
 * - 避免显式转置
 * - 调整访问模式适配转置
 */
void gemm_transpose(const Tensor& A, const Tensor& B, Tensor& C,
                    bool transpose_A, bool transpose_B,
                    const GEMMConfig& config = GEMMConfig());

//==========================================================================
// 特定优化版本的GEMM
//==========================================================================

/**
 * @brief 基准实现 - 三重循环
 *
 * @用途:
 * - 正确性验证
 * - 小矩阵 (< 16x16)
 * - 作为其他实现的参考
 */
void gemm_naive(const Tensor& A, const Tensor& B, Tensor& C);

/**
 * @brief OpenMP并行版本
 *
 * @优化点:
 * - 并行化外层循环
 * - 负载均衡
 * - 适合中大型矩阵
 */
void gemm_omp(const Tensor& A, const Tensor& B, Tensor& C);

/**
 * @brief ARM NEON优化版本 (推荐用于 ROCK 4D Cortex-A72 大核)
 *
 * @优化技术:
 * 1. **缓存分块**: 适配Cortex-A72的L1/L2缓存
 * 2. **寄存器打包**: 一次处理4x8元素块
 * 3. **FMA指令**: vmlaq_f32乘加融合
 * 4. **软件预取**: __builtin_prefetch减少stall
 * 5. **循环展开**: 4x展开隐藏延迟
 *
 * 实际性能取决于矩阵形状、线程数和 CPU 调频，应使用 benchmark 实测。
 */
void gemm_neon(const Tensor& A, const Tensor& B, Tensor& C,
               const GEMMConfig& config = GEMMConfig());

/**
 * @brief 微内核 - 4x8 GEMM
 *
 * @detail:
 * 计算 C[4][8] += A[4][K] * B[K][8]
 *
 * @优化点:
 * - 完全展开循环
 * - 数据保持在寄存器
 * - 最大化寄存器复用
 *
 * @为什么是4x8:
 * - NEON寄存器有32个128位寄存器
 * - 4个寄存器存A的行
 * - 8个寄存器存C的累加器
 * - 剩余寄存器用于B的加载和临时值
 */
void gemm_micro_kernel_4x8(
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc,
    int K);

/**
 * @brief 打包微内核 - 4x4 GEMM
 *
 * @detail:
 * 用于处理N不是8的倍数的边界情况
 */
void gemm_micro_kernel_4x4(
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc,
    int K);

//==========================================================================
// 矩阵打包 (Packing) 优化
//==========================================================================

/**
 * @brief 打包A矩阵 (M x K) 为分块格式
 *
 * @detail:
 * 将矩阵按mc x kc分块重新排列，使得:
 * - 每个块在内存中连续
 * - 适配缓存行边界
 * - 减少缓存miss
 *
 * @布局变换:
 * 原始: [M][K] 按行存储
 * 打包: [M/mc][K/kc][mc][kc] 分块存储
 *
 * @性能提升: 20-30% (取决于矩阵大小)
 */
void pack_matrix_A(const float* src, float* dst,
                   int M, int K, int mc, int kc);

/**
 * @brief 打包B矩阵 (K x N) 为分块格式
 *
 * @detail:
 * 将矩阵按kc x nc分块重新排列
 * 特别优化B矩阵的访问模式（列访问转行访问）
 *
 * @布局变换:
 * 原始: [K][N] 按行存储
 * 打包: [N/nc][K/kc][kc][nc] 分块存储
 */
void pack_matrix_B(const float* src, float* dst,
                   int K, int N, int kc, int nc);

//==========================================================================
// 工具函数
//==========================================================================

/**
 * @brief 自动选择最优GEMM实现
 *
 * @param M, N, K 矩阵维度
 * @return 推荐的实现类型
 *
 * @决策逻辑:
 * - M,N,K < 16: Naive (开销小)
 * - M,N,K < 128: NEON (缓存友好)
 * - 其他: NEON + OpenMP (并行化)
 */
GEMMImpl select_best_implementation(int M, int N, int K);

/**
 * @brief 获取最优配置
 *
 * @detail:
 * 根据 ARM64 Cortex-A72 的缓存和寄存器特性选择参数
 */
GEMMConfig get_optimal_config(int M, int N, int K);

/**
 * @brief 性能基准测试
 */
GEMMStats benchmark_gemm(int M, int N, int K,
                         std::function<void()> gemm_func,
                         int iterations = 100);

/**
 * @brief 验证GEMM结果正确性
 */
bool verify_gemm(const Tensor& C, const Tensor& C_ref, float eps = 1e-4);

//==========================================================================
// SGEMM (单精度浮点) 专用接口
//==========================================================================

/**
 * @brief 原始SGEMM接口 (兼容BLAS)
 *
 * @param layout Row-Major ('R') 或 Column-Major ('C')
 * @param transA 是否转置A
 * @param transB 是否转置B
 * @param M 矩阵A行数 / C行数
 * @param N 矩阵B列数 / C列数
 * @param K 矩阵A列数 / B行数
 * @param alpha 标量系数
 * @param A 矩阵A指针
 * @param lda A的主维度
 * @param B 矩阵B指针
 * @param ldb B的主维度
 * @param beta C的系数
 * @param C 矩阵C指针
 * @param ldc C的主维度
 */
void sgemm(char layout, char transA, char transB,
           int M, int N, int K,
           float alpha,
           const float* A, int lda,
           const float* B, int ldb,
           float beta,
           float* C, int ldc);

} // namespace microflow

#endif // MICROFLOW_GEMM_HPP
