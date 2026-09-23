#include "microflow/layers.hpp"
#include <cstring>   // 必须有，为了 std::memcpy
#include <algorithm> // 必须有，为了 std::max / std::min
#include <cmath>     // 必须有，为了 std::exp / std::tanh
#include <cstdlib>   // 必须有，为了解决之前 aligned_alloc 的潜在报错

// ARM NEON
#if defined(__aarch64__) || defined(__arm__)
    #include <arm_neon.h>
    #define MICROFLOW_HAS_NEON
#endif

// OpenMP
#include <omp.h>

namespace microflow {

//==========================================================================
// 激活函数实现
//==========================================================================

void relu(Tensor& input) {
    float* ptr = input.raw_ptr();
    uint32_t size = input.size();

#ifdef MICROFLOW_HAS_NEON
    // NEON优化的ReLU
    float32x4_t zero = vdupq_n_f32(0.0f);

    int i = 0;
    for (; i <= static_cast<int>(size) - 16; i += 16) {
        // 4x展开
        float32x4_t v0 = vld1q_f32(&ptr[i + 0]);
        float32x4_t v1 = vld1q_f32(&ptr[i + 4]);
        float32x4_t v2 = vld1q_f32(&ptr[i + 8]);
        float32x4_t v3 = vld1q_f32(&ptr[i + 12]);

        v0 = vmaxq_f32(v0, zero);
        v1 = vmaxq_f32(v1, zero);
        v2 = vmaxq_f32(v2, zero);
        v3 = vmaxq_f32(v3, zero);

        vst1q_f32(&ptr[i + 0], v0);
        vst1q_f32(&ptr[i + 4], v1);
        vst1q_f32(&ptr[i + 8], v2);
        vst1q_f32(&ptr[i + 12], v3);
    }

    for (; i <= static_cast<int>(size) - 4; i += 4) {
        float32x4_t v = vld1q_f32(&ptr[i]);
        v = vmaxq_f32(v, zero);
        vst1q_f32(&ptr[i], v);
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

void relu6(Tensor& input) {
    float* ptr = input.raw_ptr();
    uint32_t size = input.size();

#ifdef MICROFLOW_HAS_NEON
    float32x4_t zero = vdupq_n_f32(0.0f);
    float32x4_t six = vdupq_n_f32(6.0f);

    int i = 0;
    for (; i <= static_cast<int>(size) - 4; i += 4) {
        float32x4_t v = vld1q_f32(&ptr[i]);
        // clamp(v, 0, 6) = max(min(v, 6), 0)
        v = vminq_f32(v, six);
        v = vmaxq_f32(v, zero);
        vst1q_f32(&ptr[i], v);
    }

    for (; i < static_cast<int>(size); ++i) {
        ptr[i] = std::min(std::max(0.0f, ptr[i]), 6.0f);
    }
#else
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(size); ++i) {
        ptr[i] = std::min(std::max(0.0f, ptr[i]), 6.0f);
    }
#endif
}

void leaky_relu(Tensor& input, float alpha) {
    float* ptr = input.raw_ptr();
    uint32_t size = input.size();

#ifdef MICROFLOW_HAS_NEON
    float32x4_t alpha_vec = vdupq_n_f32(alpha);
    float32x4_t zero = vdupq_n_f32(0.0f);

    int i = 0;
    for (; i <= static_cast<int>(size) - 4; i += 4) {
        float32x4_t v = vld1q_f32(&ptr[i]);
        // v >= 0 ? v : alpha * v
        float32x4_t neg = vmulq_f32(v, alpha_vec);
        uint32x4_t mask = vcltq_f32(v, zero);  // v < 0
        // 选择: mask ? neg : v
        v = vbslq_f32(mask, neg, v);
        vst1q_f32(&ptr[i], v);
    }

    for (; i < static_cast<int>(size); ++i) {
        ptr[i] = ptr[i] >= 0 ? ptr[i] : alpha * ptr[i];
    }
#else
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(size); ++i) {
        ptr[i] = ptr[i] >= 0 ? ptr[i] : alpha * ptr[i];
    }
#endif
}

void elu(Tensor& input, float alpha) {
    float* ptr = input.raw_ptr();
    uint32_t size = input.size();

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(size); ++i) {
        if (ptr[i] >= 0) {
            // 保持不变
        } else {
            ptr[i] = alpha * (std::exp(ptr[i]) - 1.0f);
        }
    }
}

void gelu(Tensor& input) {
    float* ptr = input.raw_ptr();
    uint32_t size = input.size();

    // GELU近似: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(size); ++i) {
        float x = ptr[i];
        float x_cube = x * x * x;
        float tanh_arg = sqrt_2_over_pi * (x + coeff * x_cube);
        float tanh_val = std::tanh(tanh_arg);
        ptr[i] = 0.5f * x * (1.0f + tanh_val);
    }
}

void sigmoid(Tensor& input) {
    float* ptr = input.raw_ptr();
    uint32_t size = input.size();

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(size); ++i) {
        // 数值稳定的sigmoid: 1 / (1 + exp(-x))
        // 对于大x: exp(-x) 可能溢出
        // 对于小x: exp(x) 可能溢出
        float x = ptr[i];
        if (x >= 0) {
            ptr[i] = 1.0f / (1.0f + std::exp(-x));
        } else {
            float exp_x = std::exp(x);
            ptr[i] = exp_x / (1.0f + exp_x);
        }
    }
}

void tanh(Tensor& input) {
    float* ptr = input.raw_ptr();
    uint32_t size = input.size();

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(size); ++i) {
        ptr[i] = std::tanh(ptr[i]);
    }
}

void softmax(Tensor& input, int axis) {
    // 简化实现: 只支持axis=-1 (最后一个维度)
    const auto& shapes = input.shapes();

    // 计算外部维度和softmax维度
    int outer_size = 1;
    int dim_size = shapes.back();
    for (size_t i = 0; i < shapes.size() - 1; ++i) {
        outer_size *= shapes[i];
    }

    float* ptr = input.raw_ptr();

    // 对每个softmax单元计算
    #pragma omp parallel for
    for (int i = 0; i < outer_size; ++i) {
        float* row = ptr + i * dim_size;

        // 找最大值 (数值稳定性)
        float max_val = row[0];
        for (int j = 1; j < dim_size; ++j) {
            if (row[j] > max_val) {
                max_val = row[j];
            }
        }

        // 计算exp和并归一化
        float sum = 0.0f;
        for (int j = 0; j < dim_size; ++j) {
            row[j] = std::exp(row[j] - max_val);
            sum += row[j];
        }

        // 归一化
        float inv_sum = 1.0f / sum;
        for (int j = 0; j < dim_size; ++j) {
            row[j] *= inv_sum;
        }
    }
}

void log_softmax(Tensor& input, int axis) {
    const auto& shapes = input.shapes();

    int outer_size = 1;
    int dim_size = shapes.back();
    for (size_t i = 0; i < shapes.size() - 1; ++i) {
        outer_size *= shapes[i];
    }

    float* ptr = input.raw_ptr();

    #pragma omp parallel for
    for (int i = 0; i < outer_size; ++i) {
        float* row = ptr + i * dim_size;

        // 找最大值
        float max_val = row[0];
        for (int j = 1; j < dim_size; ++j) {
            if (row[j] > max_val) {
                max_val = row[j];
            }
        }

        // 计算logsumexp
        float sum_exp = 0.0f;
        for (int j = 0; j < dim_size; ++j) {
            sum_exp += std::exp(row[j] - max_val);
        }

        float log_sum = max_val + std::log(sum_exp);

        // 计算log_softmax
        for (int j = 0; j < dim_size; ++j) {
            row[j] -= log_sum;
        }
    }
}

//==========================================================================
// 池化层实现
//==========================================================================

void max_pool2d(const Tensor& input, Tensor& output,
               int kernel_size, int stride, int padding)
{
    const auto& in_shapes = input.shapes();
    int C = in_shapes[0];
    int H = in_shapes[1];
    int W = in_shapes[2];

    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;

    const float* in_ptr = input.raw_ptr();
    float* out_ptr = output.raw_ptr();

    #pragma omp parallel for collapse(2)
    for (int c = 0; c < C; ++c) {
        for (int h_out = 0; h_out < H_out; ++h_out) {
            for (int w_out = 0; w_out < W_out; ++w_out) {
                int h_start = h_out * stride - padding;
                int w_start = w_out * stride - padding;

                float max_val = -INFINITY;

                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int h = h_start + kh;
                        int w = w_start + kw;

                        if (h >= 0 && h < H && w >= 0 && w < W) {
                            float val = in_ptr[c * H * W + h * W + w];
                            if (val > max_val) {
                                max_val = val;
                            }
                        }
                    }
                }

                out_ptr[c * H_out * W_out + h_out * W_out + w_out] = max_val;
            }
        }
    }
}

void avg_pool2d(const Tensor& input, Tensor& output,
               int kernel_size, int stride, int padding)
{
    const auto& in_shapes = input.shapes();
    int C = in_shapes[0];
    int H = in_shapes[1];
    int W = in_shapes[2];

    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;

    const float* in_ptr = input.raw_ptr();
    float* out_ptr = output.raw_ptr();

    #pragma omp parallel for collapse(2)
    for (int c = 0; c < C; ++c) {
        for (int h_out = 0; h_out < H_out; ++h_out) {
            for (int w_out = 0; w_out < W_out; ++w_out) {
                int h_start = h_out * stride - padding;
                int w_start = w_out * stride - padding;

                float sum = 0.0f;
                int count = 0;

                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int h = h_start + kh;
                        int w = w_start + kw;

                        if (h >= 0 && h < H && w >= 0 && w < W) {
                            sum += in_ptr[c * H * W + h * W + w];
                            ++count;
                        }
                    }
                }

                out_ptr[c * H_out * W_out + h_out * W_out + w_out] =
                    sum / count;
            }
        }
    }
}

void global_avg_pool2d(const Tensor& input, Tensor& output) {
    const auto& in_shapes = input.shapes();
    int C = in_shapes[0];
    int H = in_shapes[1];
    int W = in_shapes[2];

    const float* in_ptr = input.raw_ptr();
    float* out_ptr = output.raw_ptr();

    float inv_hw = 1.0f / (H * W);

    #pragma omp parallel for
    for (int c = 0; c < C; ++c) {
        float sum = 0.0f;
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                sum += in_ptr[c * H * W + h * W + w];
            }
        }
        out_ptr[c] = sum * inv_hw;
    }
}

void adaptive_avg_pool2d(const Tensor& input, Tensor& output,
                        int output_height, int output_width)
{
    const auto& in_shapes = input.shapes();
    int C = in_shapes[0];
    int H = in_shapes[1];
    int W = in_shapes[2];

    const float* in_ptr = input.raw_ptr();
    float* out_ptr = output.raw_ptr();

    // 计算步长和核大小
    int stride_h = (H - 1) / output_height + 1;
    int stride_w = (W - 1) / output_width + 1;
    // 简化实现: 使用固定步长
    #pragma omp parallel for collapse(3)
    for (int c = 0; c < C; ++c) {
        for (int h_out = 0; h_out < output_height; ++h_out) {
            for (int w_out = 0; w_out < output_width; ++w_out) {
                // 计算输入区域
                int h_start = h_out * stride_h;
                int w_start = w_out * stride_w;

                float sum = 0.0f;
                int count = 0;

                for (int kh = 0; kh < stride_h && h_start + kh < H; ++kh) {
                    for (int kw = 0; kw < stride_w && w_start + kw < W; ++kw) {
                        sum += in_ptr[c * H * W + (h_start + kh) * W + (w_start + kw)];
                        ++count;
                    }
                }

                out_ptr[c * output_height * output_width + h_out * output_width + w_out] =
                    sum / count;
            }
        }
    }
}

//==========================================================================
// 归一化层实现
//==========================================================================

void batch_norm(Tensor& input,
               const Tensor& mean,
               const Tensor& var,
               const Tensor& gamma,
               const Tensor& beta,
               float eps)
{
    const auto& shapes = input.shapes();
    int C = shapes[0];
    int S = 1;  // 每个通道的元素数
    for (size_t i = 1; i < shapes.size(); ++i) {
        S *= shapes[i];
    }

    float* in_ptr = input.raw_ptr();
    const float* mean_ptr = mean.raw_ptr();
    const float* var_ptr = var.raw_ptr();
    const float* gamma_ptr = gamma.raw_ptr();
    const float* beta_ptr = beta.raw_ptr();

    // 预计算缩放和偏移
    std::vector<float> scale(C);
    std::vector<float> offset(C);

    for (int c = 0; c < C; ++c) {
        float std_dev = 1.0f / std::sqrt(var_ptr[c] + eps);
        scale[c] = gamma_ptr[c] * std_dev;
        offset[c] = beta_ptr[c] - mean_ptr[c] * scale[c];
    }

    // 应用归一化
    #pragma omp parallel for
    for (int i = 0; i < C * S; ++i) {
        int c = i / S;
        in_ptr[i] = in_ptr[i] * scale[c] + offset[c];
    }
}

void layer_norm(Tensor& input,
               const Tensor& gamma,
               const Tensor& beta,
               float eps,
               int axis)
{
    // 简化实现: axis=-1
    const auto& shapes = input.shapes();
    int outer_size = 1;
    int inner_size = shapes.back();
    for (size_t i = 0; i < shapes.size() - 1; ++i) {
        outer_size *= shapes[i];
    }

    float* in_ptr = input.raw_ptr();
    const float* gamma_ptr = gamma.raw_ptr();
    const float* beta_ptr = beta.raw_ptr();

    #pragma omp parallel for
    for (int i = 0; i < outer_size; ++i) {
        float* row = in_ptr + i * inner_size;

        // 计算均值
        float mean = 0.0f;
        for (int j = 0; j < inner_size; ++j) {
            mean += row[j];
        }
        mean /= inner_size;

        // 计算方差
        float var = 0.0f;
        for (int j = 0; j < inner_size; ++j) {
            float diff = row[j] - mean;
            var += diff * diff;
        }
        var /= inner_size;

        float std_dev = 1.0f / std::sqrt(var + eps);

        // 归一化
        for (int j = 0; j < inner_size; ++j) {
            row[j] = (row[j] - mean) * std_dev * gamma_ptr[j] + beta_ptr[j];
        }
    }
}

void group_norm(Tensor& input,
               const Tensor& gamma,
               const Tensor& beta,
               int num_groups,
               float eps)
{
    // 简化实现
    // 实际实现需要按组计算均值和方差
    batch_norm(input, Tensor(), Tensor(), gamma, beta, eps);
}

//==========================================================================
// 全连接层实现
//==========================================================================

void linear(const Tensor& input,
           const Tensor& weight,
           const Tensor& bias,
           Tensor& output)
{
    // Y = X * W^T + bias
    // 优化版本：权重已在导出时转置 [in_features, out_features]
    // 向后兼容：自动检测并支持原始格式

    if (!input.is_valid() || !weight.is_valid()) {
        std::cerr << "ERROR: Invalid input or weight tensor in linear!\n";
        return;
    }

    // 对于 1D 输入
    if (input.ndim() == 1) {
        int in_features = input.shapes()[0];
        int out_features;

        // 检测权重格式：通过比较第一个维度和输入特征数
        bool is_transposed = (weight.shapes()[0] == static_cast<uint32_t>(in_features));

        const Tensor* weight_ptr;
        Tensor weight_transposed;

        if (is_transposed) {
            // 优化格式: [in_features, out_features]
            out_features = weight.shapes()[1];
            weight_ptr = &weight;
        } else {
            // 原始格式: [out_features, in_features]
            out_features = weight.shapes()[0];
            weight_transposed = weight.transpose(0, 1);
            weight_ptr = &weight_transposed;
        }

        Tensor input_2d = input.reshape({1, static_cast<uint32_t>(in_features)});
        Tensor output_2d = output.reshape({1, static_cast<uint32_t>(out_features)});

        gemm(input_2d, *weight_ptr, output_2d);

        if (bias.size() > 0 && bias.is_valid()) {
            float* out_ptr = output_2d.raw_ptr();
            const float* bias_ptr = bias.raw_ptr();
            for (int n = 0; n < out_features; ++n) {
                out_ptr[n] += bias_ptr[n];
            }
        }
    } else if (input.ndim() == 2) {
        int in_features = input.shapes()[1];

        bool is_transposed = (weight.shapes()[0] == static_cast<uint32_t>(in_features));

        const Tensor* weight_ptr;
        Tensor weight_transposed;

        if (is_transposed) {
            weight_ptr = &weight;
        } else {
            weight_transposed = weight.transpose(0, 1);
            weight_ptr = &weight_transposed;
        }

        gemm(input, *weight_ptr, output);

        if (bias.size() > 0 && bias.is_valid()) {
            const auto& out_shapes = output.shapes();
            int M = out_shapes[0];
            int N = out_shapes[1];

            float* out_ptr = output.raw_ptr();
            const float* bias_ptr = bias.raw_ptr();

            #pragma omp parallel for
            for (int m = 0; m < M; ++m) {
                for (int n = 0; n < N; ++n) {
                    out_ptr[m * N + n] += bias_ptr[n];
                }
            }
        }
    } else {
        std::cerr << "ERROR: Unsupported input dimensions in linear: " << input.ndim() << "\n";
    }
}

void linear_relu(const Tensor& input,
                const Tensor& weight,
                const Tensor& bias,
                Tensor& output)
{
    linear(input, weight, bias, output);
    relu(output);
}

//==========================================================================
// 形状变换实现
//==========================================================================

void flatten(const Tensor& input, Tensor& output) {
    const float* in_ptr = input.raw_ptr();
    float* out_ptr = output.raw_ptr();
    uint32_t size = input.size();

    std::memcpy(out_ptr, in_ptr, size * sizeof(float));
}

//==========================================================================
// 拼接和分割实现
//==========================================================================

void concat(const std::vector<Tensor>& tensors, Tensor& output, int dim) {
    float* out_ptr = output.raw_ptr();
    size_t offset = 0;

    for (const auto& tensor : tensors) {
        const float* in_ptr = tensor.raw_ptr();
        size_t copy_size = tensor.size() * sizeof(float);
        std::memcpy(out_ptr + offset, in_ptr, copy_size);
        offset += tensor.size();
    }
}

void split(const Tensor& input, std::vector<Tensor>& outputs,
          int dim, const std::vector<int>& split_sizes)
{
    const float* in_ptr = input.raw_ptr();
    size_t offset = 0;

    for (size_t i = 0; i < outputs.size(); ++i) {
        float* out_ptr = outputs[i].raw_ptr();
        size_t copy_size = outputs[i].size() * sizeof(float);
        std::memcpy(out_ptr, in_ptr + offset, copy_size);
        offset += outputs[i].size();
    }
}

void stack(const std::vector<Tensor>& tensors, Tensor& output, int dim) {
    // 简化实现: dim=0
    float* out_ptr = output.raw_ptr();
    size_t offset = 0;

    for (const auto& tensor : tensors) {
        const float* in_ptr = tensor.raw_ptr();
        size_t copy_size = tensor.size() * sizeof(float);
        std::memcpy(out_ptr + offset, in_ptr, copy_size);
        offset += tensor.size();
    }
}

//==========================================================================
// 上采样和下采样实现
//==========================================================================

void upsample_bilinear(const Tensor& input, Tensor& output,
                      float scale_factor)
{
    const auto& in_shapes = input.shapes();
    int C = in_shapes[0];
    int H = in_shapes[1];
    int W = in_shapes[2];

    int H_out = static_cast<int>(H * scale_factor);
    int W_out = static_cast<int>(W * scale_factor);

    const float* in_ptr = input.raw_ptr();
    float* out_ptr = output.raw_ptr();

    float scale_h = static_cast<float>(H) / H_out;
    float scale_w = static_cast<float>(W) / W_out;

    #pragma omp parallel for collapse(3)
    for (int c = 0; c < C; ++c) {
        for (int h_out = 0; h_out < H_out; ++h_out) {
            for (int w_out = 0; w_out < W_out; ++w_out) {
                float h_in = h_out * scale_h;
                float w_in = w_out * scale_w;

                int h0 = static_cast<int>(h_in);
                int w0 = static_cast<int>(w_in);
                int h1 = std::min(h0 + 1, H - 1);
                int w1 = std::min(w0 + 1, W - 1);

                float dh = h_in - h0;
                float dw = w_in - w0;

                // 双线性插值
                float v00 = in_ptr[c * H * W + h0 * W + w0];
                float v01 = in_ptr[c * H * W + h0 * W + w1];
                float v10 = in_ptr[c * H * W + h1 * W + w0];
                float v11 = in_ptr[c * H * W + h1 * W + w1];

                float v0 = v00 * (1 - dw) + v01 * dw;
                float v1 = v10 * (1 - dw) + v11 * dw;
                float val = v0 * (1 - dh) + v1 * dh;

                out_ptr[c * H_out * W_out + h_out * W_out + w_out] = val;
            }
        }
    }
}

void upsample_nearest(const Tensor& input, Tensor& output,
                     int scale_factor)
{
    const auto& in_shapes = input.shapes();
    int C = in_shapes[0];
    int H = in_shapes[1];
    int W = in_shapes[2];

    int H_out = H * scale_factor;
    int W_out = W * scale_factor;

    const float* in_ptr = input.raw_ptr();
    float* out_ptr = output.raw_ptr();

    #pragma omp parallel for collapse(3)
    for (int c = 0; c < C; ++c) {
        for (int h_out = 0; h_out < H_out; ++h_out) {
            for (int w_out = 0; w_out < W_out; ++w_out) {
                int h_in = h_out / scale_factor;
                int w_in = w_out / scale_factor;

                out_ptr[c * H_out * W_out + h_out * W_out + w_out] =
                    in_ptr[c * H * W + h_in * W + w_in];
            }
        }
    }
}

void downsample(const Tensor& input, Tensor& output, int stride) {
    const auto& in_shapes = input.shapes();
    int C = in_shapes[0];
    int H = in_shapes[1];
    int W = in_shapes[2];

    int H_out = H / stride;
    int W_out = W / stride;

    const float* in_ptr = input.raw_ptr();
    float* out_ptr = output.raw_ptr();

    #pragma omp parallel for collapse(3)
    for (int c = 0; c < C; ++c) {
        for (int h_out = 0; h_out < H_out; ++h_out) {
            for (int w_out = 0; w_out < W_out; ++w_out) {
                int h_in = h_out * stride;
                int w_in = w_out * stride;

                out_ptr[c * H_out * W_out + h_out * W_out + w_out] =
                    in_ptr[c * H * W + h_in * W + w_in];
            }
        }
    }
}

//==========================================================================
// 填充和裁剪实现
//==========================================================================

void pad2d(const Tensor& input, Tensor& output,
          int pad_h, int pad_w, float value)
{
    const auto& in_shapes = input.shapes();
    int C = in_shapes[0];
    int H = in_shapes[1];
    int W = in_shapes[2];

    int H_out = H + 2 * pad_h;
    int W_out = W + 2 * pad_w;

    const float* in_ptr = input.raw_ptr();
    float* out_ptr = output.raw_ptr();

    // 先填充指定值
    output.fill(value);

    // 复制输入数据
    #pragma omp parallel for collapse(3)
    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                out_ptr[c * H_out * W_out + (h + pad_h) * W_out + (w + pad_w)] =
                    in_ptr[c * H * W + h * W + w];
            }
        }
    }
}

void crop2d(const Tensor& input, Tensor& output,
           int crop_h, int crop_w)
{
    const auto& in_shapes = input.shapes();
    int C = in_shapes[0];
    int H = in_shapes[1];
    int W = in_shapes[2];

    int H_out = H - 2 * crop_h;
    int W_out = W - 2 * crop_w;

    const float* in_ptr = input.raw_ptr();
    float* out_ptr = output.raw_ptr();

    #pragma omp parallel for collapse(3)
    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H_out; ++h) {
            for (int w = 0; w < W_out; ++w) {
                out_ptr[c * H_out * W_out + h * W_out + w] =
                    in_ptr[c * H * W + (h + crop_h) * W + (w + crop_w)];
            }
        }
    }
}

//==========================================================================
// 工具函数实现
//==========================================================================

Tensor clone(const Tensor& input) {
    Tensor result(input.shapes(), input.layout());
    input.copy_to(result.raw_ptr());
    return result;
}

void fill(Tensor& input, float value) {
    input.fill(value);
}

Tensor eye(int n) {
    Tensor result({static_cast<uint32_t>(n), static_cast<uint32_t>(n)});
    result.fill(0.0f);

    float* ptr = result.raw_ptr();
    for (int i = 0; i < n; ++i) {
        ptr[i * n + i] = 1.0f;
    }

    return result;
}

Tensor diag(const Tensor& input) {
    // 简化实现: 假设输入是1D的
    int n = input.size();
    Tensor result({static_cast<uint32_t>(n), static_cast<uint32_t>(n)});
    result.fill(0.0f);

    const float* in_ptr = input.raw_ptr();
    float* out_ptr = result.raw_ptr();

    for (int i = 0; i < n; ++i) {
        out_ptr[i * n + i] = in_ptr[i];
    }

    return result;
}

} // namespace microflow
