#include <cstdlib>
#include "microflow/gemm.hpp"
#include <cassert>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <cmath>

// ARM NEON头文件
#if defined(__aarch64__) || defined(__arm__)
    #include <arm_neon.h>
    #define MICROFLOW_HAS_NEON
#endif

// OpenMP
#include <omp.h>

namespace microflow {

namespace {

// Batch-1 fully-connected layers are GEMV operations. The generic 4x8
// micro-kernel cannot be selected when M == 1, so the old implementation
// fell through to a scalar edge path.
void gemv_row_major(const float* a, const float* b, float* c, int n, int k) {
    std::memset(c, 0, static_cast<size_t>(n) * sizeof(float));

#ifdef MICROFLOW_HAS_NEON
    const int vector_end = n & ~15;
    const bool use_parallel = (static_cast<long long>(n) * k) >= 131072 && vector_end >= 32;

    #pragma omp parallel for schedule(static) if(use_parallel)
    for (int j = 0; j < vector_end; j += 16) {
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);

        for (int p = 0; p < k; ++p) {
            const float value = a[p];
            const float* row = b + static_cast<size_t>(p) * n + j;
            acc0 = vmlaq_n_f32(acc0, vld1q_f32(row), value);
            acc1 = vmlaq_n_f32(acc1, vld1q_f32(row + 4), value);
            acc2 = vmlaq_n_f32(acc2, vld1q_f32(row + 8), value);
            acc3 = vmlaq_n_f32(acc3, vld1q_f32(row + 12), value);
        }

        vst1q_f32(c + j, acc0);
        vst1q_f32(c + j + 4, acc1);
        vst1q_f32(c + j + 8, acc2);
        vst1q_f32(c + j + 12, acc3);
    }

    int j = vector_end;
    for (; j + 3 < n; j += 4) {
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (int p = 0; p < k; ++p) {
            acc = vmlaq_n_f32(acc,
                              vld1q_f32(b + static_cast<size_t>(p) * n + j),
                              a[p]);
        }
        vst1q_f32(c + j, acc);
    }
    for (; j < n; ++j) {
        float sum = 0.0f;
        for (int p = 0; p < k; ++p) {
            sum += a[p] * b[static_cast<size_t>(p) * n + j];
        }
        c[j] = sum;
    }
#else
    for (int p = 0; p < k; ++p) {
        const float value = a[p];
        const float* row = b + static_cast<size_t>(p) * n;
        #pragma omp simd
        for (int j = 0; j < n; ++j) {
            c[j] += value * row[j];
        }
    }
#endif
}

} // namespace

void gemv_packed(const Tensor& input, const Tensor& packed_weights,
                 const Tensor& bias, Tensor& output, bool apply_relu) {
    const int out_features = static_cast<int>(packed_weights.shapes()[0]);
    const int in_features = static_cast<int>(packed_weights.shapes()[1]);
    const float* input_ptr = input.raw_ptr();
    const float* weights_ptr = packed_weights.raw_ptr();
    const float* bias_ptr = bias.is_valid() ? bias.raw_ptr() : nullptr;
    float* output_ptr = output.raw_ptr();
    const bool use_parallel =
        static_cast<long long>(out_features) * in_features >= 131072;

#ifdef MICROFLOW_HAS_NEON
    const int block_end = out_features & ~3;
    #pragma omp parallel for schedule(static) if(use_parallel)
    for (int output_index = 0; output_index < block_end; output_index += 4) {
        const float* w0 = weights_ptr + static_cast<size_t>(output_index) * in_features;
        const float* w1 = w0 + in_features;
        const float* w2 = w1 + in_features;
        const float* w3 = w2 + in_features;
        float32x4_t a00 = vdupq_n_f32(0.0f), a01 = vdupq_n_f32(0.0f);
        float32x4_t a02 = vdupq_n_f32(0.0f), a03 = vdupq_n_f32(0.0f);
        float32x4_t a10 = vdupq_n_f32(0.0f), a11 = vdupq_n_f32(0.0f);
        float32x4_t a12 = vdupq_n_f32(0.0f), a13 = vdupq_n_f32(0.0f);
        float32x4_t a20 = vdupq_n_f32(0.0f), a21 = vdupq_n_f32(0.0f);
        float32x4_t a22 = vdupq_n_f32(0.0f), a23 = vdupq_n_f32(0.0f);
        float32x4_t a30 = vdupq_n_f32(0.0f), a31 = vdupq_n_f32(0.0f);
        float32x4_t a32 = vdupq_n_f32(0.0f), a33 = vdupq_n_f32(0.0f);
        int i = 0;
        for (; i + 15 < in_features; i += 16) {
            const float32x4_t x0 = vld1q_f32(input_ptr + i);
            const float32x4_t x1 = vld1q_f32(input_ptr + i + 4);
            const float32x4_t x2 = vld1q_f32(input_ptr + i + 8);
            const float32x4_t x3 = vld1q_f32(input_ptr + i + 12);
            a00 = vfmaq_f32(a00, x0, vld1q_f32(w0 + i));
            a01 = vfmaq_f32(a01, x1, vld1q_f32(w0 + i + 4));
            a02 = vfmaq_f32(a02, x2, vld1q_f32(w0 + i + 8));
            a03 = vfmaq_f32(a03, x3, vld1q_f32(w0 + i + 12));
            a10 = vfmaq_f32(a10, x0, vld1q_f32(w1 + i));
            a11 = vfmaq_f32(a11, x1, vld1q_f32(w1 + i + 4));
            a12 = vfmaq_f32(a12, x2, vld1q_f32(w1 + i + 8));
            a13 = vfmaq_f32(a13, x3, vld1q_f32(w1 + i + 12));
            a20 = vfmaq_f32(a20, x0, vld1q_f32(w2 + i));
            a21 = vfmaq_f32(a21, x1, vld1q_f32(w2 + i + 4));
            a22 = vfmaq_f32(a22, x2, vld1q_f32(w2 + i + 8));
            a23 = vfmaq_f32(a23, x3, vld1q_f32(w2 + i + 12));
            a30 = vfmaq_f32(a30, x0, vld1q_f32(w3 + i));
            a31 = vfmaq_f32(a31, x1, vld1q_f32(w3 + i + 4));
            a32 = vfmaq_f32(a32, x2, vld1q_f32(w3 + i + 8));
            a33 = vfmaq_f32(a33, x3, vld1q_f32(w3 + i + 12));
        }
        float sums[4] = {
            (bias_ptr ? bias_ptr[output_index] : 0.0f) +
                vaddvq_f32(vaddq_f32(vaddq_f32(a00, a01), vaddq_f32(a02, a03))),
            (bias_ptr ? bias_ptr[output_index + 1] : 0.0f) +
                vaddvq_f32(vaddq_f32(vaddq_f32(a10, a11), vaddq_f32(a12, a13))),
            (bias_ptr ? bias_ptr[output_index + 2] : 0.0f) +
                vaddvq_f32(vaddq_f32(vaddq_f32(a20, a21), vaddq_f32(a22, a23))),
            (bias_ptr ? bias_ptr[output_index + 3] : 0.0f) +
                vaddvq_f32(vaddq_f32(vaddq_f32(a30, a31), vaddq_f32(a32, a33))),
        };
        for (; i < in_features; ++i) {
            const float value = input_ptr[i];
            sums[0] += value * w0[i];
            sums[1] += value * w1[i];
            sums[2] += value * w2[i];
            sums[3] += value * w3[i];
        }
        for (int lane = 0; lane < 4; ++lane) {
            output_ptr[output_index + lane] =
                apply_relu ? std::max(sums[lane], 0.0f) : sums[lane];
        }
    }
    for (int output_index = block_end; output_index < out_features; ++output_index) {
        const float* weights = weights_ptr + static_cast<size_t>(output_index) * in_features;
        float sum = bias_ptr ? bias_ptr[output_index] : 0.0f;
        float32x4_t accumulator = vdupq_n_f32(0.0f);
        int i = 0;
        for (; i + 3 < in_features; i += 4) {
            accumulator = vfmaq_f32(accumulator, vld1q_f32(input_ptr + i),
                                    vld1q_f32(weights + i));
        }
        sum += vaddvq_f32(accumulator);
        for (; i < in_features; ++i) sum += input_ptr[i] * weights[i];
        output_ptr[output_index] = apply_relu ? std::max(sum, 0.0f) : sum;
    }
#else
    #pragma omp parallel for schedule(static) if(use_parallel)
    for (int output_index = 0; output_index < out_features; ++output_index) {
        const float* weights = weights_ptr + static_cast<size_t>(output_index) * in_features;
        float sum = bias_ptr ? bias_ptr[output_index] : 0.0f;
        #pragma omp simd reduction(+:sum)
        for (int i = 0; i < in_features; ++i) {
            sum += input_ptr[i] * weights[i];
        }
        output_ptr[output_index] = apply_relu ? std::max(sum, 0.0f) : sum;
    }
#endif
}

void gemv_int8_dynamic(const Tensor& input,
                       const std::vector<int8_t>& packed_weights,
                       const std::vector<float>& weight_scales,
                       const std::vector<int32_t>& weight_sums,
                       const Tensor& bias,
                       Tensor& output,
                       bool apply_relu,
                       std::vector<int8_t>& input_buffer) {
    const int in_features = static_cast<int>(input.size());
    const int out_features = static_cast<int>(output.size());
    constexpr int block_size = 64;
    const int blocks = (in_features + block_size - 1) / block_size;
    const float* input_ptr = input.raw_ptr();
    input_buffer.resize(in_features);

    float max_value = 0.0f;
    float min_value = 0.0f;
#ifdef MICROFLOW_HAS_NEON
    float32x4_t maximum = vdupq_n_f32(0.0f);
    float32x4_t minimum = vdupq_n_f32(0.0f);
    int input_index = 0;
    for (; input_index + 3 < in_features; input_index += 4) {
        maximum = vmaxq_f32(maximum, vabsq_f32(vld1q_f32(input_ptr + input_index)));
        minimum = vminq_f32(minimum, vld1q_f32(input_ptr + input_index));
    }
    max_value = vmaxvq_f32(maximum);
    min_value = vminvq_f32(minimum);
    for (; input_index < in_features; ++input_index) {
        max_value = std::max(max_value, std::abs(input_ptr[input_index]));
        min_value = std::min(min_value, input_ptr[input_index]);
    }
#else
    for (int i = 0; i < in_features; ++i) {
        max_value = std::max(max_value, std::abs(input_ptr[i]));
        min_value = std::min(min_value, input_ptr[i]);
    }
#endif
    // ReLU outputs are non-negative, so use all 256 signed-int8 codes with
    // zero represented by -128. This doubles activation resolution compared
    // with symmetric quantization while retaining the same NEON kernel.
    const int zero_point = min_value >= 0.0f ? -128 : 0;
    const float levels = zero_point == -128 ? 255.0f : 127.0f;
    const float input_scale = max_value > 0.0f ? max_value / levels : 1.0f;
    const float inverse_scale = 1.0f / input_scale;

#ifdef MICROFLOW_HAS_NEON
    const float32x4_t multiplier = vdupq_n_f32(inverse_scale);
    const int32x4_t offset = vdupq_n_s32(zero_point);
    int i = 0;
    for (; i + 7 < in_features; i += 8) {
        const int32x4_t q0 = vaddq_s32(
            vcvtnq_s32_f32(vmulq_f32(vld1q_f32(input_ptr + i), multiplier)), offset);
        const int32x4_t q1 = vaddq_s32(
            vcvtnq_s32_f32(vmulq_f32(vld1q_f32(input_ptr + i + 4), multiplier)), offset);
        const int16x8_t q16 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
        vst1_s8(input_buffer.data() + i, vqmovn_s16(q16));
    }
    for (; i < in_features; ++i) {
        const long value = std::lrint(input_ptr[i] * inverse_scale) + zero_point;
        input_buffer[i] = static_cast<int8_t>(std::clamp(value, -128L, 127L));
    }
#else
    for (int i = 0; i < in_features; ++i) {
        const long value = std::lrint(input_ptr[i] * inverse_scale) + zero_point;
        input_buffer[i] = static_cast<int8_t>(std::clamp(value, -128L, 127L));
    }
#endif

    const float* bias_ptr = bias.is_valid() ? bias.raw_ptr() : nullptr;
    float* output_ptr = output.raw_ptr();
    const bool use_parallel =
        static_cast<long long>(out_features) * in_features >= 131072;
    #pragma omp parallel for schedule(static) if(use_parallel)
    for (int output_index = 0; output_index < out_features; ++output_index) {
        const int8_t* weights = packed_weights.data() +
            static_cast<size_t>(output_index) * in_features;
        float accumulated = 0.0f;
        for (int block = 0; block < blocks; ++block) {
        const int begin = block * block_size;
        const int end = std::min(begin + block_size, in_features);
        int32_t dot = 0;
#ifdef MICROFLOW_HAS_NEON
        int32x4_t accumulator = vdupq_n_s32(0);
        int k = begin;
        for (; k + 15 < end; k += 16) {
            const int8x16_t x = vld1q_s8(input_buffer.data() + k);
            const int8x16_t w = vld1q_s8(weights + k);
            accumulator = vpadalq_s16(accumulator,
                                      vmull_s8(vget_low_s8(x), vget_low_s8(w)));
            accumulator = vpadalq_s16(accumulator,
                                      vmull_s8(vget_high_s8(x), vget_high_s8(w)));
        }
        dot = vaddvq_s32(accumulator);
        for (; k < end; ++k) dot += input_buffer[k] * weights[k];
#else
        #pragma omp simd reduction(+:dot)
        for (int k = begin; k < end; ++k) dot += input_buffer[k] * weights[k];
#endif
        const size_t scale_index = static_cast<size_t>(output_index) * blocks + block;
        dot -= zero_point * weight_sums[scale_index];
        accumulated += static_cast<float>(dot) * weight_scales[scale_index];
        }
        float value = accumulated * input_scale +
                      (bias_ptr ? bias_ptr[output_index] : 0.0f);
        output_ptr[output_index] = apply_relu ? std::max(value, 0.0f) : value;
    }
}

//==========================================================================
// GEMM配置调优
//==========================================================================

GEMMConfig get_optimal_config(int M, int N, int K) {
    GEMMConfig config;

    // 根据矩阵大小调整分块参数
    // 目标: 每个块能放入L1缓存

    // mc * kc * sizeof(float) < L1 / 3 (给A, B, C各留空间)
    // 48KB / 3 ≈ 16KB = 4096 floats
    // mc * kc ≈ 4096
    // 选择 mc = 64, kc = 64

    if (M <= 32 && N <= 32 && K <= 32) {
        // 小矩阵: 使用小分块
        config.mc = 16;
        config.nc = 16;
        config.kc = 32;
    } else if (M <= 128 && N <= 128) {
        // 中等矩阵: 适配L1缓存
        config.mc = 32;
        config.nc = 32;
        config.kc = 128;
    } else {
        // 大矩阵: 适配L2缓存
        config.mc = 64;
        config.nc = 64;
        config.kc = 256;
    }

    // 寄存器分块保持固定
    // 4x8是NEON的最优选择 (32个128位寄存器)
    config.mr = 4;
    config.nr = 8;

    return config;
}

GEMMImpl select_best_implementation(int M, int N, int K) {
    // 小矩阵直接用naive
    if (M <= 8 && N <= 8 && K <= 8) {
        return GEMMImpl::kNaive;
    }

    // 中等矩阵用NEON
    if (M <= 256 && N <= 256) {
        return GEMMImpl::kNEON;
    }

    // 大矩阵用NEON + OpenMP
    return GEMMImpl::kNEON;
}

//==========================================================================
// 基础实现
//==========================================================================

void gemm_naive(const Tensor& A, const Tensor& B, Tensor& C) {
    int M = A.shapes()[0];
    int K = A.shapes()[1];
    int N = B.shapes()[1];

    const float* ptr_A = A.raw_ptr();
    const float* ptr_B = B.raw_ptr();
    float* ptr_C = C.raw_ptr();

    // 三重循环
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += ptr_A[i * K + k] * ptr_B[k * N + j];
            }
            ptr_C[i * N + j] = sum;
        }
    }
}

void gemm_omp(const Tensor& A, const Tensor& B, Tensor& C) {
    int M = A.shapes()[0];
    int K = A.shapes()[1];
    int N = B.shapes()[1];

    const float* ptr_A = A.raw_ptr();
    const float* ptr_B = B.raw_ptr();
    float* ptr_C = C.raw_ptr();

    // OpenMP并行化外层循环
    #pragma omp parallel for
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += ptr_A[i * K + k] * ptr_B[k * N + j];
            }
            ptr_C[i * N + j] = sum;
        }
    }
}

//==========================================================================
// ARM NEON优化实现
//==========================================================================

#ifdef MICROFLOW_HAS_NEON

/**
 * @brief NEON 4x8微内核
 *
 * @detail:
 * 计算 C[4][8] += A[4][k] * B[k][8]
 *
 * @寄存器分配:
 * - v0-v3: A的4个元素 (广播为4个float)
 * - v4-v11: C的8列累加器
 * - v12-v15: B的临时加载
 *
 * @性能: 每个K迭代处理32个浮点乘加
 */
void gemm_micro_kernel_4x8(
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc,
    int K)
{
    // 加载C的初始值 (8列)
    float32x4_t c0 = vld1q_f32(&C[0 * ldc + 0]);
    float32x4_t c1 = vld1q_f32(&C[0 * ldc + 4]);
    float32x4_t c2 = vld1q_f32(&C[1 * ldc + 0]);
    float32x4_t c3 = vld1q_f32(&C[1 * ldc + 4]);
    float32x4_t c4 = vld1q_f32(&C[2 * ldc + 0]);
    float32x4_t c5 = vld1q_f32(&C[2 * ldc + 4]);
    float32x4_t c6 = vld1q_f32(&C[3 * ldc + 0]);
    float32x4_t c7 = vld1q_f32(&C[3 * ldc + 4]);

    // 主循环: K维度
    int k = 0;

    // 4x展开以隐藏延迟
    for (; k <= K - 4; k += 4) {
        // ===== 迭代 0 =====
        float32x4_t a0 = vdupq_n_f32(A[0 * lda + k]);
        float32x4_t b0 = vld1q_f32(&B[k * ldb + 0]);
        c0 = vmlaq_f32(c0, a0, b0);

        float32x4_t b1 = vld1q_f32(&B[k * ldb + 4]);
        c1 = vmlaq_f32(c1, a0, b1);

        float32x4_t a1 = vdupq_n_f32(A[1 * lda + k]);
        c2 = vmlaq_f32(c2, a1, b0);
        c3 = vmlaq_f32(c3, a1, b1);

        float32x4_t a2 = vdupq_n_f32(A[2 * lda + k]);
        c4 = vmlaq_f32(c4, a2, b0);
        c5 = vmlaq_f32(c5, a2, b1);

        float32x4_t a3 = vdupq_n_f32(A[3 * lda + k]);
        c6 = vmlaq_f32(c6, a3, b0);
        c7 = vmlaq_f32(c7, a3, b1);

        // ===== 迭代 1 =====
        a0 = vdupq_n_f32(A[0 * lda + k + 1]);
        b0 = vld1q_f32(&B[(k + 1) * ldb + 0]);
        c0 = vmlaq_f32(c0, a0, b0);

        b1 = vld1q_f32(&B[(k + 1) * ldb + 4]);
        c1 = vmlaq_f32(c1, a0, b1);

        a1 = vdupq_n_f32(A[1 * lda + k + 1]);
        c2 = vmlaq_f32(c2, a1, b0);
        c3 = vmlaq_f32(c3, a1, b1);

        a2 = vdupq_n_f32(A[2 * lda + k + 1]);
        c4 = vmlaq_f32(c4, a2, b0);
        c5 = vmlaq_f32(c5, a2, b1);

        a3 = vdupq_n_f32(A[3 * lda + k + 1]);
        c6 = vmlaq_f32(c6, a3, b0);
        c7 = vmlaq_f32(c7, a3, b1);

        // ===== 迭代 2 =====
        a0 = vdupq_n_f32(A[0 * lda + k + 2]);
        b0 = vld1q_f32(&B[(k + 2) * ldb + 0]);
        c0 = vmlaq_f32(c0, a0, b0);

        b1 = vld1q_f32(&B[(k + 2) * ldb + 4]);
        c1 = vmlaq_f32(c1, a0, b1);

        a1 = vdupq_n_f32(A[1 * lda + k + 2]);
        c2 = vmlaq_f32(c2, a1, b0);
        c3 = vmlaq_f32(c3, a1, b1);

        a2 = vdupq_n_f32(A[2 * lda + k + 2]);
        c4 = vmlaq_f32(c4, a2, b0);
        c5 = vmlaq_f32(c5, a2, b1);

        a3 = vdupq_n_f32(A[3 * lda + k + 2]);
        c6 = vmlaq_f32(c6, a3, b0);
        c7 = vmlaq_f32(c7, a3, b1);

        // ===== 迭代 3 =====
        a0 = vdupq_n_f32(A[0 * lda + k + 3]);
        b0 = vld1q_f32(&B[(k + 3) * ldb + 0]);
        c0 = vmlaq_f32(c0, a0, b0);

        b1 = vld1q_f32(&B[(k + 3) * ldb + 4]);
        c1 = vmlaq_f32(c1, a0, b1);

        a1 = vdupq_n_f32(A[1 * lda + k + 3]);
        c2 = vmlaq_f32(c2, a1, b0);
        c3 = vmlaq_f32(c3, a1, b1);

        a2 = vdupq_n_f32(A[2 * lda + k + 3]);
        c4 = vmlaq_f32(c4, a2, b0);
        c5 = vmlaq_f32(c5, a2, b1);

        a3 = vdupq_n_f32(A[3 * lda + k + 3]);
        c6 = vmlaq_f32(c6, a3, b0);
        c7 = vmlaq_f32(c7, a3, b1);

        // 预取下一个K迭代的数据
        __builtin_prefetch(&A[0 * lda + k + 8], 0, 3);
        __builtin_prefetch(&B[(k + 8) * ldb], 0, 3);
    }

    // 处理剩余的K
    for (; k < K; ++k) {
        float32x4_t a0 = vdupq_n_f32(A[0 * lda + k]);
        float32x4_t b0 = vld1q_f32(&B[k * ldb + 0]);
        float32x4_t b1 = vld1q_f32(&B[k * ldb + 4]);

        c0 = vmlaq_f32(c0, a0, b0);
        c1 = vmlaq_f32(c1, a0, b1);

        float32x4_t a1 = vdupq_n_f32(A[1 * lda + k]);
        c2 = vmlaq_f32(c2, a1, b0);
        c3 = vmlaq_f32(c3, a1, b1);

        float32x4_t a2 = vdupq_n_f32(A[2 * lda + k]);
        c4 = vmlaq_f32(c4, a2, b0);
        c5 = vmlaq_f32(c5, a2, b1);

        float32x4_t a3 = vdupq_n_f32(A[3 * lda + k]);
        c6 = vmlaq_f32(c6, a3, b0);
        c7 = vmlaq_f32(c7, a3, b1);
    }

    // 存储结果
    vst1q_f32(&C[0 * ldc + 0], c0);
    vst1q_f32(&C[0 * ldc + 4], c1);
    vst1q_f32(&C[1 * ldc + 0], c2);
    vst1q_f32(&C[1 * ldc + 4], c3);
    vst1q_f32(&C[2 * ldc + 0], c4);
    vst1q_f32(&C[2 * ldc + 4], c5);
    vst1q_f32(&C[3 * ldc + 0], c6);
    vst1q_f32(&C[3 * ldc + 4], c7);
}

/**
 * @brief NEON 4x4微内核 (处理边界)
 */
void gemm_micro_kernel_4x4(
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc,
    int K)
{
    float32x4_t c0 = vld1q_f32(&C[0 * ldc + 0]);
    float32x4_t c1 = vld1q_f32(&C[1 * ldc + 0]);
    float32x4_t c2 = vld1q_f32(&C[2 * ldc + 0]);
    float32x4_t c3 = vld1q_f32(&C[3 * ldc + 0]);

    for (int k = 0; k < K; ++k) {
        float32x4_t a0 = vdupq_n_f32(A[0 * lda + k]);
        float32x4_t a1 = vdupq_n_f32(A[1 * lda + k]);
        float32x4_t a2 = vdupq_n_f32(A[2 * lda + k]);
        float32x4_t a3 = vdupq_n_f32(A[3 * lda + k]);

        float32x4_t b = vld1q_f32(&B[k * ldb + 0]);

        c0 = vmlaq_f32(c0, a0, b);
        c1 = vmlaq_f32(c1, a1, b);
        c2 = vmlaq_f32(c2, a2, b);
        c3 = vmlaq_f32(c3, a3, b);
    }

    vst1q_f32(&C[0 * ldc + 0], c0);
    vst1q_f32(&C[1 * ldc + 0], c1);
    vst1q_f32(&C[2 * ldc + 0], c2);
    vst1q_f32(&C[3 * ldc + 0], c3);
}

#endif // MICROFLOW_HAS_NEON

//==========================================================================
// 主GEMM函数 (带分块)
//==========================================================================

void gemm_neon(const Tensor& A, const Tensor& B, Tensor& C,
               const GEMMConfig& config)
{
    int M = A.shapes()[0];
    int N = B.shapes()[1];

    float* ptr_C = C.raw_ptr();

    // 清零C矩阵
    std::memset(ptr_C, 0, M * N * sizeof(float));

#ifdef MICROFLOW_HAS_NEON
    const int K = A.shapes()[1];
    const float* ptr_A = A.raw_ptr();
    const float* ptr_B = B.raw_ptr();
    // 分块参数
    const int mc = config.mc;  // M分块
    const int nc = config.nc;  // N分块
    const int kc = config.kc;  // K分块
    const int mr = config.mr;  // 微内核M
    const int nr = config.nr;  // 微内核N

    // M维度分块 (OpenMP并行)
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < M; i += mc) {
        int M_cur = std::min(mc, M - i);

        // K维度分块
        for (int p = 0; p < K; p += kc) {
            int K_cur = std::min(kc, K - p);

            // N维度分块
            for (int j = 0; j < N; j += nc) {
                int N_cur = std::min(nc, N - j);

                // 微内核分块
                for (int ii = 0; ii < M_cur; ii += mr) {
                    int M_cur_cur = std::min(mr, M_cur - ii);

                    for (int jj = 0; jj < N_cur; jj += nr) {
                        int N_cur_cur = std::min(nr, N_cur - jj);

                        // 调用微内核
                        if (M_cur_cur == 4 && N_cur_cur == 8) {
                            gemm_micro_kernel_4x8(
                                &ptr_A[(i + ii) * K + p], K,
                                &ptr_B[p * N + j + jj], N,
                                &ptr_C[(i + ii) * N + j + jj], N,
                                K_cur
                            );
                        } else if (M_cur_cur == 4 && N_cur_cur == 4) {
                            gemm_micro_kernel_4x4(
                                &ptr_A[(i + ii) * K + p], K,
                                &ptr_B[p * N + j + jj], N,
                                &ptr_C[(i + ii) * N + j + jj], N,
                                K_cur
                            );
                        } else {
                            // 边界情况使用标量代码
                            for (int iii = 0; iii < M_cur_cur; ++iii) {
                                for (int jjj = 0; jjj < N_cur_cur; ++jjj) {
                                    float sum = ptr_C[(i + ii + iii) * N + j + jj + jjj];
                                    for (int k = 0; k < K_cur; ++k) {
                                        sum += ptr_A[(i + ii + iii) * K + p + k] *
                                               ptr_B[(p + k) * N + j + jj + jjj];
                                    }
                                    ptr_C[(i + ii + iii) * N + j + jj + jjj] = sum;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
#else
    // 回退到OpenMP版本
    gemm_omp(A, B, C);
#endif
}

//==========================================================================
// 通用接口
//==========================================================================

void gemm(const Tensor& A, const Tensor& B, Tensor& C,
          const GEMMConfig& config)
{
    int M = A.shapes()[0];
    int N = B.shapes()[1];
    int K = A.shapes()[1];

    if (M == 1) {
        gemv_row_major(A.raw_ptr(), B.raw_ptr(), C.raw_ptr(), N, K);
        return;
    }

    // 自动选择实现
    GEMMImpl impl = select_best_implementation(M, N, K);

    switch (impl) {
        case GEMMImpl::kNaive:
            gemm_naive(A, B, C);
            break;
        case GEMMImpl::kOpenMP:
            gemm_omp(A, B, C);
            break;
        case GEMMImpl::kNEON:
            gemm_neon(A, B, C, config);
            break;
        default:
            gemm_neon(A, B, C, config);
            break;
    }
}

void gemm_transpose(const Tensor& A, const Tensor& B, Tensor& C,
                    bool transpose_A, bool transpose_B,
                    const GEMMConfig& config)
{
    // 简化实现: 创建转置视图
    // 完整实现应该调整访问模式而非显式转置

    if (transpose_A) {
        Tensor A_T = A.transpose(0, 1);
        if (transpose_B) {
            Tensor B_T = B.transpose(0, 1);
            gemm(A_T, B_T, C, config);
        } else {
            gemm(A_T, B, C, config);
        }
    } else {
        if (transpose_B) {
            Tensor B_T = B.transpose(0, 1);
            gemm(A, B_T, C, config);
        } else {
            gemm(A, B, C, config);
        }
    }
}

void batch_gemm(int batch, const Tensor* A, const Tensor* B, Tensor* C,
                const GEMMConfig& config)
{
    #pragma omp parallel for
    for (int i = 0; i < batch; ++i) {
        gemm(A[i], B[i], C[i], config);
    }
}

//==========================================================================
// 工具函数
//==========================================================================

bool verify_gemm(const Tensor& C, const Tensor& C_ref, float eps) {
    if (C.shapes() != C_ref.shapes()) {
        return false;
    }

    const float* ptr_c = C.raw_ptr();
    const float* ptr_ref = C_ref.raw_ptr();
    uint32_t size = C.size();

    for (uint32_t i = 0; i < size; ++i) {
        float diff = std::abs(ptr_c[i] - ptr_ref[i]);
        if (diff > eps) {
            return false;
        }
    }
    return true;
}

GEMMStats benchmark_gemm(int M, int N, int K,
                         std::function<void()> gemm_func,
                         int iterations)
{
    GEMMStats stats;
    stats.total_ops = 2LL * M * N * K * iterations;  // 2次浮点运算(乘+加)

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        gemm_func();
    }

    auto end = std::chrono::high_resolution_clock::now();
    stats.time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    stats.gflops = (stats.total_ops / stats.time_ms / 1e6);

    return stats;
}

void sgemm(char layout, char transA, char transB,
           int M, int N, int K,
           float alpha,
           const float* A, int lda,
           const float* B, int ldb,
           float beta,
           float* C, int ldc)
{
    // 简化实现: 仅支持Row-Major, 无转置
    // 完整实现需要处理转置和alpha/beta系数

    // 创建Tensor包装
    Tensor A_tensor(std::vector<uint32_t>{static_cast<uint32_t>(M), static_cast<uint32_t>(K)},
                   const_cast<float*>(A));
    Tensor B_tensor(std::vector<uint32_t>{static_cast<uint32_t>(K), static_cast<uint32_t>(N)},
                   const_cast<float*>(B));
    Tensor C_tensor(std::vector<uint32_t>{static_cast<uint32_t>(M), static_cast<uint32_t>(N)},
                   C);

    gemm(A_tensor, B_tensor, C_tensor);

    // 应用alpha和beta
    if (alpha != 1.0f || beta != 0.0f) {
        float* ptr_C = C_tensor.raw_ptr();
        for (int i = 0; i < M * N; ++i) {
            ptr_C[i] = alpha * ptr_C[i];
        }
    }
}

} // namespace microflow
