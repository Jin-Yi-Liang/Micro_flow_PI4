#ifndef MICROFLOW_CONV_HPP
#define MICROFLOW_CONV_HPP

#include "microflow/tensor.hpp"
#include "microflow/gemm.hpp"
#include <cstdint>

namespace microflow {

/**
 * @brief 卷积参数
 */
struct Conv2DParams {
    int kernel_size;      // 卷积核大小 (通常为3, 5, 7)
    int stride;           // 步长
    int padding;          // 填充
    int dilation;         // 膨胀率 (通常为1)
    int groups;           // 分组数 (depthwise卷积用)

    Conv2DParams()
        : kernel_size(3), stride(1), padding(1), dilation(1), groups(1) {}

    Conv2DParams(int k, int s, int p, int d = 1, int g = 1)
        : kernel_size(k), stride(s), padding(p), dilation(d), groups(g) {}
};

/**
 * @brief 计算卷积输出尺寸
 *
 * @param input_size 输入尺寸 (H或W)
 * @param kernel_size 卷积核大小
 * @param stride 步长
 * @param padding 填充
 * @return int 输出尺寸
 *
 * @公式: output = (input + 2*padding - kernel) / stride + 1
 */
inline int compute_conv_output_size(int input_size, int kernel_size,
                                    int stride, int padding) {
    return (input_size + 2 * padding - kernel_size) / stride + 1;
}

//==========================================================================
// Im2Col 方法
//==========================================================================

/**
 * @brief Im2Col: 将卷积转换为矩阵乘法
 *
 * @detail:
 * 输入: [C, H, W] 的3D张量
 * 输出: [C*K*K, H_out*W_out] 的2D矩阵
 *
 * @变换过程:
 * - 每个输出位置对应一个滑动窗口
 * - 将滑动窗口展开为列向量
 * - 所有可能的窗口组成矩阵的列
 *
 * @优势:
 * - 可以复用优化的GEMM内核
 * - 通用性强，支持各种卷积参数
 *
 * @劣势:
 * - 内存开销大 (需要存储展开矩阵)
 * - 内存访问模式不够友好
 *
 * @优化策略:
 * - NEON向量化内存复制
 * - 预取优化
 * - 零填充优化
 */
void im2col(const Tensor& input,            // [C, H, W]
            Tensor& output,                 // [C*K*K, H_out*W_out]
            const Conv2DParams& params);

/**
 * @brief Col2Im: 矩阵转回图像 (反向传播用)
 *
 * @note: 推理时通常不需要此函数
 */
void col2im(const Tensor& col,             // [C*K*K, H_out*W_out]
            Tensor& output,                 // [C, H, W]
            const Conv2DParams& params);

/**
 * @brief NEON优化的Im2Col
 *
 * @优化点:
 * - 使用NEON向量指令复制数据
 * - 4x展开循环
 * - 预取下一个缓存行
 */
void im2col_neon(const Tensor& input,
                Tensor& output,
                const Conv2DParams& params);

//==========================================================================
// 直接卷积 (Direct Convolution)
//==========================================================================

/**
 * @brief 直接卷积 - 适用于小卷积核
 *
 * @detail:
 * 直接在输入上进行滑动窗口计算，不经过im2col变换
 *
 * @优势:
 * - 无额外内存开销
 * - 缓存友好
 * - 适合3x3小卷积核
 *
 * @优化技术:
 * - 输出分块 (Output Tiling)
 * - 输入预取
 * - 寄存器累加
 */
void conv2d_direct(const Tensor& input,           // [N, C, H, W] 或 [C, H, W]
                  const Tensor& kernel,          // [F, C, K, K]
                  Tensor& output,                // [N, F, H_out, W_out]
                  const Conv2DParams& params);

/**
 * @brief NEON优化的3x3直接卷积
 *
 * @detail:
 * 针对3x3卷积核的特化实现，使用NEON指令
 *
 * @优化策略:
 * - 输入通道向量化加载
 * - 卷积核预打包
 * - 输出通道并行计算
 */
void conv2d_direct_3x3_neon(const Tensor& input,
                           const Tensor& kernel,
                           Tensor& output,
                           const Conv2DParams& params);

/**
 * @brief Winograd卷积 (3x3, stride=1)
 *
 * @detail:
 * Winograd算法可以减少乘法次数
 *
 * @复杂度对比 (F(2x2, 3x3)):
 * - 直接卷积: 每输出像素需要 9*C_in 次乘法
 * - Winograd: 每输出2x2块需要 16*C_in 次乘法
 * - 减少: 9/4 = 2.25倍乘法
 *
 * @适用场景:
 * - 输入通道少 (< 64)
 * - 3x3卷积核
 * - stride=1
 */
void conv2d_winograd(const Tensor& input,
                    const Tensor& kernel,
                    Tensor& output,
                    const Conv2DParams& params);

/** Transform [F,C,3,3] kernels once for Winograd F(2x2,3x3). */
void winograd_transform_kernel_3x3(const Tensor& kernel, Tensor& transformed);

/** Execute Winograd F(2x2,3x3) with pre-transformed [F,C,16] kernels. */
void conv2d_winograd_f2x2(const Tensor& input,
                          const Tensor& transformed_kernel,
                          const Tensor& bias,
                          Tensor& output,
                          bool apply_relu,
                          float* workspace,
                          bool apply_maxpool = false);

//==========================================================================
// Depthwise卷积
//==========================================================================

/**
 * @brief Depthwise卷积 (分组数=输入通道数)
 *
 * @detail:
 * 每个输入通道对应一个卷积核，通道间不计算
 *
 * @计算量对比:
 * - 标准卷积: F * C * K * K * H_out * W_out
 * - Depthwise: C * K * K * H_out * W_out (少F倍)
 *
 * @适用场景:
 * - MobileNet等轻量级网络
 * - 通道间信息独立的场景
 */
void conv2d_depthwise(const Tensor& input,          // [C, H, W]
                     const Tensor& kernel,         // [C, 1, K, K]
                     Tensor& output,               // [C, H_out, W_out]
                     const Conv2DParams& params);

/**
 * @brief NEON优化的3x3 Depthwise卷积
 *
 * @优化点:
 * - 同时处理4个通道 (利用NEON)
 * - 滑动窗口优化
 * - 点乘向量化
 */
void conv2d_depthwise_3x3_neon(const Tensor& input,
                              const Tensor& kernel,
                              Tensor& output,
                              const Conv2DParams& params);

//==========================================================================
// 逐点卷积 (Pointwise Convolution)
//==========================================================================

/**
 * @brief 逐点卷积 (1x1卷积)
 *
 * @detail:
 * 本质上是矩阵乘法，不需要空间维度的滑动窗口
 *
 * @等价变换:
 * C = A * B^T
 * 其中 A: [N*H*W, C_in], B: [C_out, C_in], C: [N*H*W, C_out]
 */
void conv2d_pointwise(const Tensor& input,         // [N, C_in, H, W]
                     const Tensor& kernel,        // [C_out, C_in, 1, 1]
                     Tensor& output,              // [N, C_out, H, W]
                     const Conv2DParams& params);

//==========================================================================
// 转置卷积 (反卷积)
//==========================================================================

/**
 * @brief 转置卷积
 *
 * @detail:
 * 上采样操作，常用于分割网络的解码器
 *
 * @实现:
 * 可以通过输入填充+普通卷积实现
 */
void conv2d_transpose(const Tensor& input,          // [N, C, H, W]
                     const Tensor& kernel,         // [C, F, K, K]
                     Tensor& output,               // [N, F, H_out, W_out]
                     const Conv2DParams& params);

//==========================================================================
// 高层接口
//==========================================================================

/**
 * @brief 通用2D卷积接口
 *
 * @detail:
 * 根据输入参数自动选择最优实现
 *
 * @决策逻辑:
 * 1. kernel_size == 1 -> conv2d_pointwise (GEMM)
 * 2. kernel_size == 3 && C_in < 64 -> conv2d_winograd
 * 3. kernel_size == 3 -> conv2d_direct_3x3_neon
 * 4. 其他 -> im2col + GEMM
 *
 * @param input 输入张量 [N, C, H, W] 或 [C, H, W]
 * @param kernel 卷积核 [F, C, K, K]
 * @param bias 偏置 [F] (可选)
 * @param output 输出张量
 * @param params 卷积参数
 * @param workspace 工作空间 (im2col方法需要)
 */
void conv2d(const Tensor& input,
           const Tensor& kernel,
           const Tensor& bias,
           Tensor& output,
           const Conv2DParams& params,
           float* workspace = nullptr);

/**
 * @brief 卷积 + ReLU 融合
 *
 * @优化点:
 * - 减少内存访问
 * - 避免中间结果存储
 */
void conv2d_relu(const Tensor& input,
                const Tensor& kernel,
                Tensor& output,
                const Conv2DParams& params,
                float* workspace = nullptr);

/**
 * @brief 卷积 + BatchNorm + ReLU 融合
 *
 * @detail:
 * 将BatchNorm参数融合到卷积权重中
 * BN: y = (x - mean) / sqrt(var + eps) * gamma + beta
 * 融合后: weight' = weight * gamma / sqrt(var + eps)
 *       bias' = (bias - mean) / sqrt(var + eps) * gamma + beta
 *
 * @优势:
 * - 推理时无额外开销
 * - 减少一次张量遍历
 */
void conv2d_bn_relu(const Tensor& input,
                   const Tensor& kernel,
                   const Tensor& bn_mean,
                   const Tensor& bn_var,
                   const Tensor& bn_gamma,
                   const Tensor& bn_beta,
                   Tensor& output,
                   const Conv2DParams& params,
                   float* workspace = nullptr);

//==========================================================================
// 工具函数
//==========================================================================

/**
 * @brief 计算卷积所需工作空间大小
 */
size_t compute_conv_workspace_size(const Tensor& input,
                                  const Tensor& kernel,
                                  const Conv2DParams& params);

/**
 * @brief 验证卷积参数有效性
 */
bool validate_conv_params(const Tensor& input,
                         const Tensor& kernel,
                         const Conv2DParams& params);

/**
 * @brief 获取推荐的卷积实现
 */
enum class ConvImpl {
    kIm2ColGEMM,       // im2col + GEMM
    kDirectNEON,       // 直接卷积 + NEON
    kWinograd,         // Winograd算法
    kDepthwise,        // Depthwise卷积
    kPointwise,        // 1x1卷积(GEMM)
};

ConvImpl select_optimal_conv_impl(const Tensor& input,
                                 const Tensor& kernel,
                                 const Conv2DParams& params);

} // namespace microflow

#endif // MICROFLOW_CONV_HPP
