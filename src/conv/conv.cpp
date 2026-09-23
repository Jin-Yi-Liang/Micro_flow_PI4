#include <cstdlib>
#include "microflow/conv.hpp"
#include <cstring>
#include <algorithm>
#include <cmath>

// ARM NEON
#if defined(__aarch64__) || defined(__arm__)
    #include <arm_neon.h>
    #define MICROFLOW_HAS_NEON
#endif

// OpenMP
#include <omp.h>

namespace microflow {

//==========================================================================
// Im2Col 实现
//==========================================================================

void im2col(const Tensor& input, Tensor& output, const Conv2DParams& params) {
    // 获取输入维度
    const auto& shapes = input.shapes();
    int C = shapes[0];   // 通道数
    int H = shapes[1];   // 高度
    int W = shapes[2];   // 宽度

    int K = params.kernel_size;
    int stride = params.stride;
    int pad = params.padding;

    // 计算输出尺寸
    int H_out = (H + 2 * pad - K) / stride + 1;
    int W_out = (W + 2 * pad - K) / stride + 1;

    // 输出矩阵: [C*K*K, H_out*W_out]
    const float* in_ptr = input.raw_ptr();
    float* out_ptr = output.raw_ptr();

    // 填充im2col矩阵
    // 每一列对应一个输出位置的滑动窗口

    int col_idx = 0;
    for (int h_out = 0; h_out < H_out; ++h_out) {
        for (int w_out = 0; w_out < W_out; ++w_out) {
            // 计算输入图像的起始位置
            int h_in = h_out * stride - pad;
            int w_in = w_out * stride - pad;

            // 填充当前列 (C*K*K 个元素)
            int row_idx = 0;
            for (int c = 0; c < C; ++c) {
                for (int kh = 0; kh < K; ++kh) {
                    for (int kw = 0; kw < K; ++kw) {
                        int h = h_in + kh;
                        int w = w_in + kw;

                        // 检查边界
                        if (h >= 0 && h < H && w >= 0 && w < W) {
                            int in_idx = c * H * W + h * W + w;
                            out_ptr[row_idx * (H_out * W_out) + col_idx] =
                                in_ptr[in_idx];
                        } else {
                            // Padding区域填0
                            out_ptr[row_idx * (H_out * W_out) + col_idx] = 0.0f;
                        }
                        ++row_idx;
                    }
                }
            }
            ++col_idx;
        }
    }
}

#ifdef MICROFLOW_HAS_NEON

void im2col_neon(const Tensor& input, Tensor& output, const Conv2DParams& params) {
    const auto& shapes = input.shapes();
    int C = shapes[0];
    int H = shapes[1];
    int W = shapes[2];

    int K = params.kernel_size;
    int stride = params.stride;
    int pad = params.padding;

    int H_out = (H + 2 * pad - K) / stride + 1;
    int W_out = (W + 2 * pad - K) / stride + 1;

    const float* in_ptr = input.raw_ptr();
    float* out_ptr = output.raw_ptr();

    int total_out = H_out * W_out;

    // OpenMP并行化外层循环 (输出位置)
    #pragma omp parallel for
    for (int out_idx = 0; out_idx < total_out; ++out_idx) {
        int h_out = out_idx / W_out;
        int w_out = out_idx % W_out;

        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;

        int row_idx = 0;

        // 对每个输入通道
        for (int c = 0; c < C; ++c) {
            // 对卷积核的每个位置
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    int h = h_in + kh;
                    int w = w_in + kw;

                    float val = 0.0f;
                    if (h >= 0 && h < H && w >= 0 && w < W) {
                        val = in_ptr[c * H * W + h * W + w];
                    }

                    out_ptr[row_idx * total_out + out_idx] = val;
                    ++row_idx;
                }
            }
        }
    }
}

#endif // MICROFLOW_HAS_NEON

void col2im(const Tensor& col, Tensor& output, const Conv2DParams& params) {
    // 获取维度
    int C = output.shapes()[0];
    int H = output.shapes()[1];
    int W = output.shapes()[2];

    int K = params.kernel_size;
    int stride = params.stride;
    int pad = params.padding;

    int H_out = (H + 2 * pad - K) / stride + 1;
    int W_out = (W + 2 * pad - K) / stride + 1;

    // 清零输出
    output.fill(0.0f);

    const float* col_ptr = col.raw_ptr();
    float* out_ptr = output.raw_ptr();

    // 反向填充
    int col_idx = 0;
    for (int h_out = 0; h_out < H_out; ++h_out) {
        for (int w_out = 0; w_out < W_out; ++w_out) {
            int h_in = h_out * stride - pad;
            int w_in = w_out * stride - pad;

            int row_idx = 0;
            for (int c = 0; c < C; ++c) {
                for (int kh = 0; kh < K; ++kh) {
                    for (int kw = 0; kw < K; ++kw) {
                        int h = h_in + kh;
                        int w = w_in + kw;

                        if (h >= 0 && h < H && w >= 0 && w < W) {
                            out_ptr[c * H * W + h * W + w] +=
                                col_ptr[row_idx * (H_out * W_out) + col_idx];
                        }
                        ++row_idx;
                    }
                }
            }
            ++col_idx;
        }
    }
}

//==========================================================================
// 直接卷积实现
//==========================================================================

void conv2d_direct(const Tensor& input,
                  const Tensor& kernel,
                  Tensor& output,
                  const Conv2DParams& params)
{
    // 获取维度
    const auto& in_shapes = input.shapes();
    const auto& k_shapes = kernel.shapes();

    int C_in = in_shapes[0];
    int H = in_shapes[1];
    int W = in_shapes[2];

    int F = k_shapes[0];  // 输出通道数
    int K = params.kernel_size;
    int stride = params.stride;
    int pad = params.padding;

    int H_out = (H + 2 * pad - K) / stride + 1;
    int W_out = (W + 2 * pad - K) / stride + 1;

    const float* in_ptr = input.raw_ptr();
    const float* k_ptr = kernel.raw_ptr();
    float* out_ptr = output.raw_ptr();

    // 清零输出
    std::memset(out_ptr, 0, F * H_out * W_out * sizeof(float));

    // 三重循环: 输出通道 -> 输出位置 -> 卷积计算
    #pragma omp parallel for collapse(2)
    for (int f = 0; f < F; ++f) {
        for (int h_out = 0; h_out < H_out; ++h_out) {
            for (int w_out = 0; w_out < W_out; ++w_out) {
                int h_in = h_out * stride - pad;
                int w_in = w_out * stride - pad;

                float sum = 0.0f;

                // 对所有输入通道和卷积核位置
                for (int c = 0; c < C_in; ++c) {
                    for (int kh = 0; kh < K; ++kh) {
                        for (int kw = 0; kw < K; ++kw) {
                            int h = h_in + kh;
                            int w = w_in + kw;

                            if (h >= 0 && h < H && w >= 0 && w < W) {
                                float in_val = in_ptr[c * H * W + h * W + w];
                                float k_val = k_ptr[f * C_in * K * K +
                                                  c * K * K + kh * K + kw];
                                sum += in_val * k_val;
                            }
                        }
                    }
                }

                out_ptr[f * H_out * W_out + h_out * W_out + w_out] = sum;
            }
        }
    }
}

#ifdef MICROFLOW_HAS_NEON

void conv2d_direct_3x3_neon(const Tensor& input,
                           const Tensor& kernel,
                           Tensor& output,
                           const Conv2DParams& params)
{
    // 针对3x3卷积的NEON优化实现
    if (params.kernel_size != 3) {
        // 回退到通用实现
        conv2d_direct(input, kernel, output, params);
        return;
    }

    const auto& in_shapes = input.shapes();
    const auto& k_shapes = kernel.shapes();

    int C_in = in_shapes[0];
    int H = in_shapes[1];
    int W = in_shapes[2];

    int F = k_shapes[0];
    int K = 3;  // 固定为3x3
    int stride = params.stride;
    int pad = params.padding;

    int H_out = (H + 2 * pad - K) / stride + 1;
    int W_out = (W + 2 * pad - K) / stride + 1;

    const float* in_ptr = input.raw_ptr();
    const float* k_ptr = kernel.raw_ptr();
    float* out_ptr = output.raw_ptr();

    // 清零输出
    std::memset(out_ptr, 0, F * H_out * W_out * sizeof(float));

    // Fast path for the common 3x3/s1/p1 case. Interior pixels need no
    // bounds checks and eight neighbouring outputs are accumulated at once.
    if (stride != 1 || pad != 1 || H_out != H || W_out != W || H < 3 || W < 3) {
        conv2d_direct(input, kernel, output, params);
        return;
    }

    auto scalar_at = [=](int f, int h_out, int w_out) {
        float sum = 0.0f;
        for (int c = 0; c < C_in; ++c) {
            const float* kernel_c = k_ptr + (static_cast<size_t>(f) * C_in + c) * 9;
            for (int kh = 0; kh < 3; ++kh) {
                const int h = h_out - 1 + kh;
                if (h < 0 || h >= H) continue;
                for (int kw = 0; kw < 3; ++kw) {
                    const int w = w_out - 1 + kw;
                    if (w >= 0 && w < W) {
                        sum += in_ptr[(static_cast<size_t>(c) * H + h) * W + w] *
                               kernel_c[kh * 3 + kw];
                    }
                }
            }
        }
        return sum;
    };

    #pragma omp parallel for schedule(static)
    for (int f = 0; f < F; ++f) {
        for (int h_out = 0; h_out < H_out; ++h_out) {
            if (h_out == 0 || h_out == H_out - 1) {
                for (int w_out = 0; w_out < W_out; ++w_out) {
                    out_ptr[(static_cast<size_t>(f) * H_out + h_out) * W_out + w_out] =
                        scalar_at(f, h_out, w_out);
                }
                continue;
            }

            out_ptr[(static_cast<size_t>(f) * H_out + h_out) * W_out] =
                scalar_at(f, h_out, 0);

            int w_out = 1;
            for (; w_out + 7 <= W_out - 2; w_out += 8) {
                float32x4_t acc0 = vdupq_n_f32(0.0f);
                float32x4_t acc1 = vdupq_n_f32(0.0f);
                for (int c = 0; c < C_in; ++c) {
                    const float* kernel_c = k_ptr + (static_cast<size_t>(f) * C_in + c) * 9;
                    for (int kh = 0; kh < 3; ++kh) {
                        const float* row = in_ptr +
                            (static_cast<size_t>(c) * H + (h_out - 1 + kh)) * W + w_out - 1;
                        acc0 = vmlaq_n_f32(acc0, vld1q_f32(row), kernel_c[kh * 3]);
                        acc0 = vmlaq_n_f32(acc0, vld1q_f32(row + 1), kernel_c[kh * 3 + 1]);
                        acc0 = vmlaq_n_f32(acc0, vld1q_f32(row + 2), kernel_c[kh * 3 + 2]);
                        acc1 = vmlaq_n_f32(acc1, vld1q_f32(row + 4), kernel_c[kh * 3]);
                        acc1 = vmlaq_n_f32(acc1, vld1q_f32(row + 5), kernel_c[kh * 3 + 1]);
                        acc1 = vmlaq_n_f32(acc1, vld1q_f32(row + 6), kernel_c[kh * 3 + 2]);
                    }
                }
                float* dst = out_ptr +
                    (static_cast<size_t>(f) * H_out + h_out) * W_out + w_out;
                vst1q_f32(dst, acc0);
                vst1q_f32(dst + 4, acc1);
            }

            for (; w_out + 3 <= W_out - 2; w_out += 4) {
                float32x4_t acc = vdupq_n_f32(0.0f);
                for (int c = 0; c < C_in; ++c) {
                    const float* kernel_c = k_ptr + (static_cast<size_t>(f) * C_in + c) * 9;
                    for (int kh = 0; kh < 3; ++kh) {
                        const float* row = in_ptr +
                            (static_cast<size_t>(c) * H + (h_out - 1 + kh)) * W + w_out - 1;
                        acc = vmlaq_n_f32(acc, vld1q_f32(row), kernel_c[kh * 3]);
                        acc = vmlaq_n_f32(acc, vld1q_f32(row + 1), kernel_c[kh * 3 + 1]);
                        acc = vmlaq_n_f32(acc, vld1q_f32(row + 2), kernel_c[kh * 3 + 2]);
                    }
                }
                vst1q_f32(out_ptr +
                          (static_cast<size_t>(f) * H_out + h_out) * W_out + w_out,
                          acc);
            }

            for (; w_out < W_out; ++w_out) {
                out_ptr[(static_cast<size_t>(f) * H_out + h_out) * W_out + w_out] =
                    scalar_at(f, h_out, w_out);
            }
        }
    }
}

#endif // MICROFLOW_HAS_NEON

//==========================================================================
// Winograd卷积 (F(2x2, 3x3))
//==========================================================================

void conv2d_winograd(const Tensor& input,
                    const Tensor& kernel,
                    Tensor& output,
                    const Conv2DParams& params)
{
    // Winograd F(2x2, 3x3) 原始实现
    // 仅支持 stride=1, padding=1 的情况

    if (params.kernel_size != 3 || params.stride != 1 || params.padding != 1) {
        conv2d_direct(input, kernel, output, params);
        return;
    }

    // Winograd变换矩阵
    // B = [[1, 0, -1],
    //      [0, 1, 1],
    //      [0, -1, 1],
    //      [0, 1, 0]]

    // 简化实现: 对于每个4x4的输入块, 计算2x2的输出块
    // 每个输出块需要16次乘法 (相比直接卷积的36次)

    // 这里使用简化版本, 实际应该使用完整的Winograd算法

    // 回退到 NEON 优化版本。
    #ifdef MICROFLOW_HAS_NEON
        conv2d_direct_3x3_neon(input, kernel, output, params);
    #else
        conv2d_direct(input, kernel, output, params);
    #endif
}

void winograd_transform_kernel_3x3(const Tensor& kernel, Tensor& transformed) {
    const int filters = static_cast<int>(kernel.shapes()[0]);
    const int channels = static_cast<int>(kernel.shapes()[1]);
    const float* source = kernel.raw_ptr();
    float* destination = transformed.raw_ptr();

    #pragma omp parallel for schedule(static)
    for (int filter = 0; filter < filters; ++filter) {
        for (int channel = 0; channel < channels; ++channel) {
            const float* g = source +
                (static_cast<size_t>(filter) * channels + channel) * 9;
            float temporary[4][3];
            for (int column = 0; column < 3; ++column) {
                temporary[0][column] = g[column];
                temporary[1][column] = 0.5f * (g[column] + g[3 + column] + g[6 + column]);
                temporary[2][column] = 0.5f * (g[column] - g[3 + column] + g[6 + column]);
                temporary[3][column] = g[6 + column];
            }
            float* u = destination +
                (static_cast<size_t>(filter) * channels + channel) * 16;
            for (int row = 0; row < 4; ++row) {
                u[row * 4] = temporary[row][0];
                u[row * 4 + 1] = 0.5f * (temporary[row][0] + temporary[row][1] +
                                         temporary[row][2]);
                u[row * 4 + 2] = 0.5f * (temporary[row][0] - temporary[row][1] +
                                         temporary[row][2]);
                u[row * 4 + 3] = temporary[row][2];
            }
        }
    }
}

void conv2d_winograd_f2x2(const Tensor& input,
                          const Tensor& transformed_kernel,
                          const Tensor& bias,
                          Tensor& output,
                          bool apply_relu,
                          float* workspace,
                          bool apply_maxpool) {
    const int channels = static_cast<int>(input.shapes()[0]);
    const int height = static_cast<int>(input.shapes()[1]);
    const int width = static_cast<int>(input.shapes()[2]);
    const int filters = static_cast<int>(transformed_kernel.shapes()[0]);
    const int output_height = static_cast<int>(output.shapes()[1]);
    const int output_width = static_cast<int>(output.shapes()[2]);
    const int convolution_height = apply_maxpool ? output_height * 2 : output_height;
    const int convolution_width = apply_maxpool ? output_width * 2 : output_width;
    const int tile_rows = (convolution_height + 1) / 2;
    const int tile_columns = (convolution_width + 1) / 2;
    const int tile_count = tile_rows * tile_columns;
    const float* input_ptr = input.raw_ptr();
    const float* kernel_ptr = transformed_kernel.raw_ptr();
    const float* bias_ptr = bias.is_valid() ? bias.raw_ptr() : nullptr;
    float* output_ptr = output.raw_ptr();

    // V = B^T d B. Layout is [tile, channel, 16], allowing all filters
    // to reuse the input transform.
    #pragma omp parallel
    {
    #pragma omp for collapse(2) schedule(static)
    for (int tile = 0; tile < tile_count; ++tile) {
        for (int channel = 0; channel < channels; ++channel) {
            const int tile_row = tile / tile_columns;
            const int tile_column = tile % tile_columns;
            const int input_row = tile_row * 2 - 1;
            const int input_column = tile_column * 2 - 1;
            float d[4][4]{};
            for (int row = 0; row < 4; ++row) {
                const int source_row = input_row + row;
                if (source_row < 0 || source_row >= height) continue;
                for (int column = 0; column < 4; ++column) {
                    const int source_column = input_column + column;
                    if (source_column >= 0 && source_column < width) {
                        d[row][column] = input_ptr[
                            (static_cast<size_t>(channel) * height + source_row) * width +
                            source_column];
                    }
                }
            }

            float temporary[4][4];
            float* v = workspace +
                (static_cast<size_t>(tile) * channels + channel) * 16;
            for (int column = 0; column < 4; ++column) {
                temporary[0][column] = d[0][column] - d[2][column];
                temporary[1][column] = d[1][column] + d[2][column];
                temporary[2][column] = -d[1][column] + d[2][column];
                temporary[3][column] = d[1][column] - d[3][column];
            }
            for (int row = 0; row < 4; ++row) {
                v[row * 4] = temporary[row][0] - temporary[row][2];
                v[row * 4 + 1] = temporary[row][1] + temporary[row][2];
                v[row * 4 + 2] = -temporary[row][1] + temporary[row][2];
                v[row * 4 + 3] = temporary[row][1] - temporary[row][3];
            }
        }
    }

    #pragma omp for schedule(static)
    for (int filter = 0; filter < filters; ++filter) {
        for (int tile = 0; tile < tile_count; ++tile) {
            float m[16]{};
#ifdef MICROFLOW_HAS_NEON
            float32x4_t m0 = vdupq_n_f32(0.0f);
            float32x4_t m1 = vdupq_n_f32(0.0f);
            float32x4_t m2 = vdupq_n_f32(0.0f);
            float32x4_t m3 = vdupq_n_f32(0.0f);
            for (int channel = 0; channel < channels; ++channel) {
                const float* u = kernel_ptr +
                    (static_cast<size_t>(filter) * channels + channel) * 16;
                const float* v = workspace +
                    (static_cast<size_t>(tile) * channels + channel) * 16;
                m0 = vfmaq_f32(m0, vld1q_f32(u), vld1q_f32(v));
                m1 = vfmaq_f32(m1, vld1q_f32(u + 4), vld1q_f32(v + 4));
                m2 = vfmaq_f32(m2, vld1q_f32(u + 8), vld1q_f32(v + 8));
                m3 = vfmaq_f32(m3, vld1q_f32(u + 12), vld1q_f32(v + 12));
            }
            vst1q_f32(m, m0);
            vst1q_f32(m + 4, m1);
            vst1q_f32(m + 8, m2);
            vst1q_f32(m + 12, m3);
#else
            for (int channel = 0; channel < channels; ++channel) {
                const float* u = kernel_ptr +
                    (static_cast<size_t>(filter) * channels + channel) * 16;
                const float* v = workspace +
                    (static_cast<size_t>(tile) * channels + channel) * 16;
                for (int i = 0; i < 16; ++i) m[i] += u[i] * v[i];
            }
#endif
            float temporary[2][4];
            for (int column = 0; column < 4; ++column) {
                temporary[0][column] = m[column] + m[4 + column] + m[8 + column];
                temporary[1][column] = m[4 + column] - m[8 + column] - m[12 + column];
            }
            float result[4] = {
                temporary[0][0] + temporary[0][1] + temporary[0][2],
                temporary[0][1] - temporary[0][2] - temporary[0][3],
                temporary[1][0] + temporary[1][1] + temporary[1][2],
                temporary[1][1] - temporary[1][2] - temporary[1][3],
            };
            const float bias_value = bias_ptr ? bias_ptr[filter] : 0.0f;
            const int output_row = (tile / tile_columns) * 2;
            const int output_column = (tile % tile_columns) * 2;
            if (apply_maxpool) {
                float value = std::max(std::max(result[0], result[1]),
                                       std::max(result[2], result[3])) + bias_value;
                if (apply_relu) value = std::max(value, 0.0f);
                output_ptr[(static_cast<size_t>(filter) * output_height +
                            tile / tile_columns) * output_width + tile % tile_columns] = value;
                continue;
            }
            for (int row = 0; row < 2; ++row) {
                if (output_row + row >= convolution_height) continue;
                for (int column = 0; column < 2; ++column) {
                    if (output_column + column >= convolution_width) continue;
                    float value = result[row * 2 + column] + bias_value;
                    if (apply_relu) value = std::max(value, 0.0f);
                    output_ptr[(static_cast<size_t>(filter) * output_height +
                                output_row + row) * output_width + output_column + column] = value;
                }
            }
        }
    }
    }
}

//==========================================================================
// Depthwise卷积
//==========================================================================

void conv2d_depthwise(const Tensor& input,
                     const Tensor& kernel,
                     Tensor& output,
                     const Conv2DParams& params)
{
    const auto& in_shapes = input.shapes();
    int C = in_shapes[0];
    int H = in_shapes[1];
    int W = in_shapes[2];

    int K = params.kernel_size;
    int stride = params.stride;
    int pad = params.padding;

    int H_out = (H + 2 * pad - K) / stride + 1;
    int W_out = (W + 2 * pad - K) / stride + 1;

    const float* in_ptr = input.raw_ptr();
    const float* k_ptr = kernel.raw_ptr();
    float* out_ptr = output.raw_ptr();

    // 每个通道独立计算
    #pragma omp parallel for
    for (int c = 0; c < C; ++c) {
        for (int h_out = 0; h_out < H_out; ++h_out) {
            for (int w_out = 0; w_out < W_out; ++w_out) {
                int h_in = h_out * stride - pad;
                int w_in = w_out * stride - pad;

                float sum = 0.0f;
                for (int kh = 0; kh < K; ++kh) {
                    for (int kw = 0; kw < K; ++kw) {
                        int h = h_in + kh;
                        int w = w_in + kw;

                        if (h >= 0 && h < H && w >= 0 && w < W) {
                            sum += in_ptr[c * H * W + h * W + w] *
                                  k_ptr[c * K * K + kh * K + kw];
                        }
                    }
                }
                out_ptr[c * H_out * W_out + h_out * W_out + w_out] = sum;
            }
        }
    }
}

#ifdef MICROFLOW_HAS_NEON

void conv2d_depthwise_3x3_neon(const Tensor& input,
                              const Tensor& kernel,
                              Tensor& output,
                              const Conv2DParams& params)
{
    if (params.kernel_size != 3) {
        conv2d_depthwise(input, kernel, output, params);
        return;
    }

    const auto& in_shapes = input.shapes();
    int C = in_shapes[0];
    int H = in_shapes[1];
    int W = in_shapes[2];

    int K = 3;
    int stride = params.stride;
    int pad = params.padding;

    int H_out = (H + 2 * pad - K) / stride + 1;
    int W_out = (W + 2 * pad - K) / stride + 1;

    const float* in_ptr = input.raw_ptr();
    const float* k_ptr = kernel.raw_ptr();
    float* out_ptr = output.raw_ptr();

    // 并行化通道
    #pragma omp parallel for
    for (int c = 0; c < C; ++c) {
        // 当前通道的卷积核 [3, 3]
        const float* k_c = k_ptr + c * 9;

        // 对输出位置循环
        for (int h_out = 0; h_out < H_out; ++h_out) {
            int w_out = 0;

            // 4x输出位置向量化
            for (; w_out <= W_out - 4; w_out += 4) {
                float32x4_t sum = vdupq_n_f32(0.0f);

                for (int kh = 0; kh < 3; ++kh) {
                    for (int kw = 0; kw < 3; ++kw) {
                        float k_val = k_c[kh * 3 + kw];
                        float32x4_t k_vec = vdupq_n_f32(k_val);

                        float in_temp[4];
                        for (int i = 0; i < 4; ++i) {
                            int h = h_out * stride - pad + kh;
                            int w = (w_out + i) * stride - pad + kw;
                            if (h >= 0 && h < H && w >= 0 && w < W) {
                                in_temp[i] = in_ptr[c * H * W + h * W + w];
                            } else {
                                in_temp[i] = 0.0f;
                            }
                        }

                        float32x4_t in_vals = vld1q_f32(in_temp);
                        sum = vmlaq_f32(sum, in_vals, k_vec);
                    }
                }

                int out_idx = c * H_out * W_out + h_out * W_out + w_out;
                vst1q_f32(&out_ptr[out_idx], sum);
            }

            // 剩余位置
            for (; w_out < W_out; ++w_out) {
                int h_in = h_out * stride - pad;
                int w_in = w_out * stride - pad;

                float sum = 0.0f;
                for (int kh = 0; kh < 3; ++kh) {
                    for (int kw = 0; kw < 3; ++kw) {
                        int h = h_in + kh;
                        int w = w_in + kw;
                        if (h >= 0 && h < H && w >= 0 && w < W) {
                            sum += in_ptr[c * H * W + h * W + w] * k_c[kh * 3 + kw];
                        }
                    }
                }
                out_ptr[c * H_out * W_out + h_out * W_out + w_out] = sum;
            }
        }
    }
}

#endif // MICROFLOW_HAS_NEON

//==========================================================================
// 逐点卷积 (1x1)
//==========================================================================

void conv2d_pointwise(const Tensor& input,
                     const Tensor& kernel,
                     Tensor& output,
                     const Conv2DParams& params)
{
    // 1x1卷积本质上是矩阵乘法
    // 输入: [C_in, H, W] -> reshape为 [H*W, C_in]
    // 卷积核: [C_out, C_in, 1, 1] -> reshape为 [C_out, C_in]
    // 输出: [C_out, H, W] -> reshape为 [H*W, C_out]

    const auto& in_shapes = input.shapes();
    const auto& k_shapes = kernel.shapes();

    int C_in = in_shapes[0];
    int H = in_shapes[1];
    int W = in_shapes[2];
    int C_out = k_shapes[0];

    const float* in_ptr = input.raw_ptr();
    const float* k_ptr = kernel.raw_ptr();
    float* out_ptr = output.raw_ptr();

    // 使用GEMM计算
    // 每个输出位置是 C_in 维向量的点积

    #pragma omp parallel for
    for (int hw = 0; hw < H * W; ++hw) {
        for (int c_out = 0; c_out < C_out; ++c_out) {
            float sum = 0.0f;
            for (int c_in = 0; c_in < C_in; ++c_in) {
                sum += in_ptr[c_in * H * W + hw] *
                       k_ptr[c_out * C_in + c_in];
            }
            out_ptr[c_out * H * W + hw] = sum;
        }
    }
}

//==========================================================================
// 转置卷积
//==========================================================================

void conv2d_transpose(const Tensor& input,
                     const Tensor& kernel,
                     Tensor& output,
                     const Conv2DParams& params)
{
    // 转置卷积可以通过输入填充+普通卷积实现
    // 简化实现: 直接使用col2im + im2col

    // 实际实现会更复杂
    // 这里暂时回退到简单实现

    const auto& in_shapes = input.shapes();
    int C_in = in_shapes[0];
    int H_in = in_shapes[1];
    int W_in = in_shapes[2];

    const auto& k_shapes = kernel.shapes();
    int C_out = k_shapes[1];  // 转置: 输入通道变成输出通道
    int K = params.kernel_size;

    // 输出尺寸计算
    int H_out = (H_in - 1) * params.stride - 2 * params.padding + (K - 1) + 1;
    int W_out = (W_in - 1) * params.stride - 2 * params.padding + (K - 1) + 1;

    // 清零输出
    output.fill(0.0f);

    const float* in_ptr = input.raw_ptr();
    const float* k_ptr = kernel.raw_ptr();
    float* out_ptr = output.raw_ptr();

    #pragma omp parallel for
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int h_in = 0; h_in < H_in; ++h_in) {
            for (int w_in = 0; w_in < W_in; ++w_in) {
                float in_val = in_ptr[c_in * H_in * W_in + h_in * W_in + w_in];

                int h_out_start = h_in * params.stride - params.padding;
                int w_out_start = w_in * params.stride - params.padding;

                for (int c_out = 0; c_out < C_out; ++c_out) {
                    for (int kh = 0; kh < K; ++kh) {
                        for (int kw = 0; kw < K; ++kw) {
                            int h_out = h_out_start + kh;
                            int w_out = w_out_start + kw;

                            if (h_out >= 0 && h_out < H_out && w_out >= 0 && w_out < W_out) {
                                float k_val = k_ptr[c_in * C_out * K * K +
                                                   c_out * K * K + kh * K + kw];
                                #pragma omp atomic
                                out_ptr[c_out * H_out * W_out + h_out * W_out + w_out] +=
                                    in_val * k_val;
                            }
                        }
                    }
                }
            }
        }
    }
}

//==========================================================================
// 高层接口
//==========================================================================

void conv2d(const Tensor& input,
           const Tensor& kernel,
           const Tensor& bias,
           Tensor& output,
           const Conv2DParams& params,
           float* workspace)
{
    // 根据参数选择最优实现
    ConvImpl impl = select_optimal_conv_impl(input, kernel, params);

    switch (impl) {
        case ConvImpl::kPointwise:
            conv2d_pointwise(input, kernel, output, params);
            break;

        case ConvImpl::kDepthwise:
            #ifdef MICROFLOW_HAS_NEON
                if (params.kernel_size == 3) {
                    conv2d_depthwise_3x3_neon(input, kernel, output, params);
                } else {
                    conv2d_depthwise(input, kernel, output, params);
                }
            #else
                conv2d_depthwise(input, kernel, output, params);
            #endif
            break;

        case ConvImpl::kDirectNEON:
            #ifdef MICROFLOW_HAS_NEON
                if (params.kernel_size == 3) {
                    conv2d_direct_3x3_neon(input, kernel, output, params);
                } else {
                    conv2d_direct(input, kernel, output, params);
                }
            #else
                conv2d_direct(input, kernel, output, params);
            #endif
            break;

        case ConvImpl::kIm2ColGEMM:
        default: {
            // 使用im2col + GEMM
            const auto& in_shapes = input.shapes();
            const auto& k_shapes = kernel.shapes();

            int C_in = in_shapes[0];
            int H = in_shapes[1];
            int W = in_shapes[2];
            int F = k_shapes[0];
            int K = params.kernel_size;

            int H_out = (H + 2 * params.padding - K) / params.stride + 1;
            int W_out = (W + 2 * params.padding - K) / params.stride + 1;

            // im2col变换
            Tensor col({static_cast<uint32_t>(C_in * K * K),
                        static_cast<uint32_t>(H_out * W_out)});
            #ifdef MICROFLOW_HAS_NEON
                im2col_neon(input, col, params);
            #else
                im2col(input, col, params);
            #endif

            // 矩阵乘法
            // kernel: [F, C_in*K*K]
            // col: [C_in*K*K, H_out*W_out]
            // output: [F, H_out*W_out]

            Tensor kernel_mat({static_cast<uint32_t>(F),
                               static_cast<uint32_t>(C_in * K * K)},
                            const_cast<float*>(kernel.raw_ptr()));
            Tensor output_mat({static_cast<uint32_t>(F),
                               static_cast<uint32_t>(H_out * W_out)},
                            output.raw_ptr());

            gemm(kernel_mat, col, output_mat);
            break;
        }
    }

    // 添加偏置
    if (bias.size() > 0) {
        const auto& out_shapes = output.shapes();
        int C_out = out_shapes[0];
        int H_out = out_shapes[1];
        int W_out = out_shapes[2];

        const float* bias_ptr = bias.raw_ptr();
        float* out_ptr = output.raw_ptr();

        #pragma omp parallel for
        for (int c = 0; c < C_out; ++c) {
            for (int i = 0; i < H_out * W_out; ++i) {
                out_ptr[c * H_out * W_out + i] += bias_ptr[c];
            }
        }
    }
}

void conv2d_relu(const Tensor& input,
                const Tensor& kernel,
                Tensor& output,
                const Conv2DParams& params,
                float* workspace)
{
    // 空偏置
    Tensor empty_bias;
    conv2d(input, kernel, empty_bias, output, params, workspace);

    // 就地ReLU
    float* ptr = output.raw_ptr();
    uint32_t size = output.size();

    #ifdef MICROFLOW_HAS_NEON
        // NEON优化的ReLU
        int i = 0;
        float32x4_t zero = vdupq_n_f32(0.0f);

        for (; i <= static_cast<int>(size) - 4; i += 4) {
            float32x4_t vals = vld1q_f32(&ptr[i]);
            // max(vals, 0)
            vals = vmaxq_f32(vals, zero);
            vst1q_f32(&ptr[i], vals);
        }

        // 处理剩余元素
        for (; i < static_cast<int>(size); ++i) {
            ptr[i] = std::max(0.0f, ptr[i]);
        }
    #else
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(size); ++i) {
            ptr[i] = std::max(0.0f, ptr[i]);
        }
    #endif
}

void conv2d_bn_relu(const Tensor& input,
                   const Tensor& kernel,
                   const Tensor& bn_mean,
                   const Tensor& bn_var,
                   const Tensor& bn_gamma,
                   const Tensor& bn_beta,
                   Tensor& output,
                   const Conv2DParams& params,
                   float* workspace)
{
    // 融合BatchNorm参数到卷积权重
    // weight' = weight * gamma / sqrt(var + eps)
    // bias' = (bias - mean) / sqrt(var + eps) * gamma + beta

    const auto& k_shapes = kernel.shapes();
    int F = k_shapes[0];
    int C_in = k_shapes[1];
    int K = k_shapes[2];

    // 创建融合后的权重
    Tensor kernel_fused({static_cast<uint32_t>(F), static_cast<uint32_t>(C_in),
                         static_cast<uint32_t>(K), static_cast<uint32_t>(K)});
    Tensor bias_fused({static_cast<uint32_t>(F)});

    const float* k_ptr = kernel.raw_ptr();
    const float* mean_ptr = bn_mean.raw_ptr();
    const float* var_ptr = bn_var.raw_ptr();
    const float* gamma_ptr = bn_gamma.raw_ptr();
    const float* beta_ptr = bn_beta.raw_ptr();

    float* k_fused_ptr = kernel_fused.raw_ptr();
    float* b_fused_ptr = bias_fused.raw_ptr();

    float eps = 1e-5f;

    for (int f = 0; f < F; ++f) {
        float scale = gamma_ptr[f] / std::sqrt(var_ptr[f] + eps);
        float shift = beta_ptr[f] - mean_ptr[f] * scale;

        for (int c = 0; c < C_in; ++c) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    int idx = f * C_in * K * K + c * K * K + kh * K + kw;
                    k_fused_ptr[idx] = k_ptr[idx] * scale;
                }
            }
        }
        b_fused_ptr[f] = shift;
    }

    // 使用融合后的参数执行卷积
    conv2d(input, kernel_fused, bias_fused, output, params, workspace);

    // In-place ReLU. Calling conv2d_relu here would run the original
    // convolution a second time and overwrite the fused BatchNorm result.
    float* output_ptr = output.raw_ptr();
    const int output_size = static_cast<int>(output.size());
    #pragma omp parallel for if(output_size >= 4096)
    for (int i = 0; i < output_size; ++i) {
        output_ptr[i] = std::max(output_ptr[i], 0.0f);
    }
}

//==========================================================================
// 工具函数
//==========================================================================

size_t compute_conv_workspace_size(const Tensor& input,
                                  const Tensor& kernel,
                                  const Conv2DParams& params)
{
    const auto& in_shapes = input.shapes();
    int C_in = in_shapes[0];
    int H = in_shapes[1];
    int W = in_shapes[2];
    int K = params.kernel_size;

    int H_out = (H + 2 * params.padding - K) / params.stride + 1;
    int W_out = (W + 2 * params.padding - K) / params.stride + 1;

    // im2col需要的大小
    return C_in * K * K * H_out * W_out * sizeof(float);
}

bool validate_conv_params(const Tensor& input,
                         const Tensor& kernel,
                         const Conv2DParams& params)
{
    const auto& in_shapes = input.shapes();
    const auto& k_shapes = kernel.shapes();

    if (in_shapes.size() != 3) return false;
    if (k_shapes.size() != 4) return false;

    if (in_shapes[0] != k_shapes[1]) return false;  // 输入通道匹配
    if (k_shapes[2] != k_shapes[3]) return false;   // 方形卷积核

    return true;
}

ConvImpl select_optimal_conv_impl(const Tensor& input,
                                 const Tensor& kernel,
                                 const Conv2DParams& params)
{
    const auto& in_shapes = input.shapes();
    const auto& k_shapes = kernel.shapes();

    int C_in = in_shapes[0];
    int F = k_shapes[0];
    int K = params.kernel_size;

    // 1x1卷积 -> Pointwise
    if (K == 1) {
        return ConvImpl::kPointwise;
    }

    // Depthwise卷积
    if (params.groups == C_in && C_in == F) {
        return ConvImpl::kDepthwise;
    }

    // 3x3卷积且输入通道少 -> 直接卷积
    if (K == 3 && C_in <= 64) {
        return ConvImpl::kDirectNEON;
    }

    // 默认使用im2col + GEMM
    return ConvImpl::kIm2ColGEMM;
}

} // namespace microflow
