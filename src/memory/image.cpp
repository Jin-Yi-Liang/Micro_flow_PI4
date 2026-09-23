// 在include之前定义IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "microflow/stb_image.h"

#include "microflow/image.hpp"
#include <fstream>
#include <cstring>
#include <algorithm>
#include <cmath>

namespace microflow {

//==========================================================================
// 公共接口实现
//==========================================================================

bool Image::load(const std::string& filename,
                 Tensor& output,
                 bool grayscale)
{
    // 检查文件扩展名
    const size_t dot = filename.find_last_of('.');
    if (dot == std::string::npos) {
        std::cerr << "ERROR: Input file has no extension: " << filename << "\n";
        return false;
    }
    std::string ext = filename.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == ".bin") {
        // MNIST float32 格式
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            std::cerr << "ERROR: Cannot open file: " << filename << "\n";
            return false;
        }

        std::streamsize file_size = file.tellg();
        file.seekg(0, std::ios::beg);

        if (file_size == 3136) {  // 784 * 4 = float32 MNIST
            std::vector<float> buffer(784);
            file.read(reinterpret_cast<char*>(buffer.data()), 3136);

            output = Tensor({1, 28, 28});
            std::memcpy(output.raw_ptr(), buffer.data(), 3136);
            file.close();
            return true;
        } else if (file_size == 784) {  // uint8 MNIST
            std::vector<uint8_t> buffer(784);
            file.read(reinterpret_cast<char*>(buffer.data()), 784);

            output = Tensor({1, 28, 28});
            float* ptr = output.raw_ptr();
            for (int i = 0; i < 784; ++i) {
                ptr[i] = static_cast<float>(buffer[i]) / 255.0f;
            }
            file.close();
            return true;
        } else {
            std::cerr << "ERROR: Unknown .bin format, size: " << file_size << "\n";
            file.close();
            return false;
        }
    }
    else if (ext == ".pgm") {
        // PGM 灰度图格式
        std::vector<uint8_t> buffer;
        int width, height;
        if (!load_pgm(filename, buffer, width, height)) {
            return false;
        }

        // 创建Tensor [1, H, W]
        output = Tensor({1, static_cast<uint32_t>(height), static_cast<uint32_t>(width)});
        float* ptr = output.raw_ptr();

        // 归一化到 0-1
        for (size_t i = 0; i < buffer.size(); ++i) {
            ptr[i] = static_cast<float>(buffer[i]) / 255.0f;
        }

        return true;
    }
    else if (ext == ".ppm") {
        // PPM 彩色图格式
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "ERROR: Cannot open file: " << filename << "\n";
            return false;
        }

        char magic[3];
        file.read(magic, 2);
        magic[2] = '\0';
        file.get(); // newline

        if (strcmp(magic, "P6") != 0) {
            std::cerr << "ERROR: Not a PPM file\n";
            file.close();
            return false;
        }

        int width, height, maxval;
        file >> width >> height >> maxval;
        file.get(); // newline

        std::vector<uint8_t> buffer(width * height * 3);
        file.read(reinterpret_cast<char*>(buffer.data()), width * height * 3);
        file.close();

        if (grayscale) {
            // 转灰度
            Tensor rgb({3, static_cast<uint32_t>(height), static_cast<uint32_t>(width)});
            float* rgb_ptr = rgb.raw_ptr();
            const size_t plane = static_cast<size_t>(width) * height;
            for (size_t i = 0; i < plane; ++i) {
                rgb_ptr[i] = static_cast<float>(buffer[i * 3]) / 255.0f;
                rgb_ptr[plane + i] = static_cast<float>(buffer[i * 3 + 1]) / 255.0f;
                rgb_ptr[2 * plane + i] = static_cast<float>(buffer[i * 3 + 2]) / 255.0f;
            }

            Tensor gray;
            to_grayscale(rgb, gray);
            output = gray;
        } else {
            output = Tensor({3, static_cast<uint32_t>(height), static_cast<uint32_t>(width)});
            float* ptr = output.raw_ptr();
            for (size_t i = 0; i < buffer.size(); ++i) {
                ptr[i] = static_cast<float>(buffer[i]) / 255.0f;
            }
        }

        return true;
    }
    else if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" ||
             ext == ".bmp" || ext == ".tga" || ext == ".psd" || ext == ".gif") {
        // 使用 stb_image 加载常见图像格式
        int width, height, channels;
        unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 0);
        if (!data) {
            std::cerr << "ERROR: Failed to load image: " << filename << "\n";
            std::cerr << "stb_image error: " << stbi_failure_reason() << "\n";
            return false;
        }

        // stb_image 返回的格式是 [H, W, C]，需要转换为 [C, H, W]
        if (channels == 1) {
            // 已经是灰度图
            output = Tensor({1, static_cast<uint32_t>(height), static_cast<uint32_t>(width)});
            float* ptr = output.raw_ptr();
            for (int i = 0; i < width * height; ++i) {
                ptr[i] = static_cast<float>(data[i]) / 255.0f;
            }
        }
        else if (channels == 3 || channels == 4) {
            // RGB 或 RGBA 图像
            Tensor rgb({3, static_cast<uint32_t>(height), static_cast<uint32_t>(width)});
            float* rgb_ptr = rgb.raw_ptr();

            // stb_image 格式是行优先的 RGB/RGBA
            // 转换为 [C, H, W] 格式
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    int src_idx = (h * width + w) * channels;
                    rgb_ptr[0 * height * width + h * width + w] = static_cast<float>(data[src_idx + 0]) / 255.0f;     // R
                    rgb_ptr[1 * height * width + h * width + w] = static_cast<float>(data[src_idx + 1]) / 255.0f;     // G
                    rgb_ptr[2 * height * width + h * width + w] = static_cast<float>(data[src_idx + 2]) / 255.0f;     // B
                }
            }

            if (grayscale) {
                to_grayscale(rgb, output);
            } else {
                output = rgb;
            }
        }
        else {
            stbi_image_free(data);
            std::cerr << "ERROR: Unsupported channel count: " << channels << "\n";
            return false;
        }

        stbi_image_free(data);
        return true;
    }
    else {
        std::cerr << "ERROR: Unsupported image format: " << ext << "\n";
        std::cerr << "Supported formats: .bin, .pgm, .ppm, .jpg, .jpeg, .png, .bmp\n";
        return false;
    }
}

bool Image::load_mnist_from_memory(const float* buffer,
                                   size_t size,
                                   Tensor& output)
{
    if (size >= 784) {
        output = Tensor({1, 28, 28});
        std::memcpy(output.raw_ptr(), buffer, 784 * sizeof(float));
        return true;
    }
    return false;
}

void Image::resize(const Tensor& input,
                  Tensor& output,
                  uint32_t new_height,
                  uint32_t new_width)
{
    const auto& shapes = input.shapes();
    uint32_t channels = shapes[0];
    uint32_t height = shapes[1];
    uint32_t width = shapes[2];

    output = Tensor({channels, new_height, new_width});

    const float* in_ptr = input.raw_ptr();
    float* out_ptr = output.raw_ptr();

    resize_bilinear(in_ptr, out_ptr,
                   height, width, new_height, new_width, channels);
}

void Image::to_grayscale(const Tensor& input,
                         Tensor& output)
{
    const auto& shapes = input.shapes();
    if (shapes[0] != 3) {
        std::cerr << "ERROR: to_grayscale requires 3-channel input\n";
        return;
    }

    uint32_t height = shapes[1];
    uint32_t width = shapes[2];

    output = Tensor({1, height, width});

    const float* rgb = input.raw_ptr();
    float* gray = output.raw_ptr();

    // 标准灰度转换: 0.299*R + 0.587*G + 0.114*B
    for (uint32_t i = 0; i < height * width; ++i) {
        float r = rgb[i];
        float g = rgb[i + height * width];
        float b = rgb[i + 2 * height * width];
        gray[i] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

void Image::normalize(Tensor& image, float max_val)
{
    float* ptr = image.raw_ptr();
    uint32_t size = image.size();

    for (uint32_t i = 0; i < size; ++i) {
        ptr[i] /= max_val;
    }
}

void Image::invert(Tensor& image)
{
    float* ptr = image.raw_ptr();
    uint32_t size = image.size();

    for (uint32_t i = 0; i < size; ++i) {
        ptr[i] = 1.0f - ptr[i];
    }
}

void Image::center_crop(const Tensor& input,
                       Tensor& output,
                       uint32_t crop_height,
                       uint32_t crop_width)
{
    const auto& shapes = input.shapes();
    uint32_t channels = shapes[0];
    uint32_t height = shapes[1];
    uint32_t width = shapes[2];

    uint32_t start_h = (height - crop_height) / 2;
    uint32_t start_w = (width - crop_width) / 2;

    output = Tensor({channels, crop_height, crop_width});

    const float* in_ptr = input.raw_ptr();
    float* out_ptr = output.raw_ptr();

    for (uint32_t c = 0; c < channels; ++c) {
        for (uint32_t h = 0; h < crop_height; ++h) {
            for (uint32_t w = 0; w < crop_width; ++w) {
                uint32_t in_idx = c * height * width +
                                 (start_h + h) * width +
                                 (start_w + w);
                uint32_t out_idx = c * crop_height * crop_width +
                                  h * crop_width + w;
                out_ptr[out_idx] = in_ptr[in_idx];
            }
        }
    }
}

//==========================================================================
// 预处理功能
//==========================================================================

void Image::binarize(const Tensor& input,
                    Tensor& output,
                    float threshold)
{
    const auto& shapes = input.shapes();
    uint32_t size = input.size();

    // 复制输入到输出
    output = Tensor(shapes);
    const float* in_ptr = input.raw_ptr();
    float* out_ptr = output.raw_ptr();
    std::memcpy(out_ptr, in_ptr, size * sizeof(float));

    // 如果threshold为负，使用Otsu算法计算阈值
    if (threshold < 0) {
        // 计算直方图
        const int HIST_BINS = 256;
        int histogram[HIST_BINS] = {0};
        int total_pixels = 0;

        for (uint32_t i = 0; i < size; ++i) {
            int bin = static_cast<int>(in_ptr[i] * 255.0f);
            bin = std::max(0, std::min(255, bin));
            histogram[bin]++;
            total_pixels++;
        }

        // Otsu算法
        float sum = 0;
        for (int i = 0; i < HIST_BINS; ++i) {
            sum += i * histogram[i];
        }

        float sum_b = 0;
        int w_b = 0;
        float max_variance = 0;
        int threshold_idx = 0;

        for (int i = 0; i < HIST_BINS; ++i) {
            w_b += histogram[i];
            if (w_b == 0) continue;

            int w_f = total_pixels - w_b;
            if (w_f == 0) break;

            sum_b += i * histogram[i];
            float m_b = sum_b / w_b;
            float m_f = (sum - sum_b) / w_f;

            float variance = w_b * w_f * (m_b - m_f) * (m_b - m_f);
            if (variance > max_variance) {
                max_variance = variance;
                threshold_idx = i;
            }
        }

        threshold = threshold_idx / 255.0f;
    }

    // 应用二值化
    for (uint32_t i = 0; i < size; ++i) {
        out_ptr[i] = (in_ptr[i] >= threshold) ? 1.0f : 0.0f;
    }
}

void Image::auto_crop(const Tensor& input,
                     Tensor& output,
                     uint32_t padding,
                     float threshold)
{
    const auto& shapes = input.shapes();
    uint32_t height = shapes[shapes.size() - 2];
    uint32_t width = shapes[shapes.size() - 1];
    const float* ptr = input.raw_ptr();

    // 找到非空白区域的边界
    int top = height, bottom = 0, left = width, right = 0;

    for (uint32_t h = 0; h < height; ++h) {
        for (uint32_t w = 0; w < width; ++w) {
            float val = ptr[h * width + w];
            if (val > threshold) {
                if (static_cast<int>(h) < top) top = h;
                if (static_cast<int>(h) > bottom) bottom = h;
                if (static_cast<int>(w) < left) left = w;
                if (static_cast<int>(w) > right) right = w;
            }
        }
    }

    // 如果没找到内容，返回原图
    if (top >= bottom || left >= right) {
        output = Tensor(shapes);
        std::memcpy(output.raw_ptr(), ptr, input.size() * sizeof(float));
        return;
    }

    // 添加padding
    top = std::max(0, static_cast<int>(top) - static_cast<int>(padding));
    bottom = std::min(static_cast<int>(height) - 1, static_cast<int>(bottom) + static_cast<int>(padding));
    left = std::max(0, static_cast<int>(left) - static_cast<int>(padding));
    right = std::min(static_cast<int>(width) - 1, static_cast<int>(right) + static_cast<int>(padding));

    uint32_t crop_height = bottom - top + 1;
    uint32_t crop_width = right - left + 1;

    // 裁剪
    output = Tensor({1, crop_height, crop_width});
    float* out_ptr = output.raw_ptr();

    for (uint32_t h = 0; h < crop_height; ++h) {
        for (uint32_t w = 0; w < crop_width; ++w) {
            out_ptr[h * crop_width + w] = ptr[(top + h) * width + (left + w)];
        }
    }
}

void Image::preprocess_mnist(const Tensor& input, Tensor& output)
{
    Tensor current = input;

    // 确保是灰度图
    if (current.shapes().size() >= 3 && current.shapes()[0] == 3) {
        Tensor gray;
        to_grayscale(current, gray);
        current = gray;
    }

    const uint32_t size = current.size();
    const float* ptr = current.raw_ptr();

    // 使用图像整体统计来判断颜色格式
    // MNIST格式: 黑底(0)白字(1)，背景占大多数
    // 拍照格式: 白底(1)黑字(0)，背景占大多数

    float sum = 0;
    for (uint32_t i = 0; i < size; ++i) {
        sum += ptr[i];
    }

    float avg = sum / size;

    // 如果图像平均亮度 > 0.6，说明是白底黑字，需要反色
    // 如果图像平均亮度 < 0.4，说明是黑底白字，不需要反色
    Tensor processed;
    if (avg > 0.6f) {
        // 白底黑字，需要反色
        processed = Tensor(current.shapes());
        std::memcpy(processed.raw_ptr(), current.raw_ptr(), current.size() * sizeof(float));
        invert(processed);
    } else {
        // 黑底白字，直接使用
        processed = current;
    }

    // 自动裁剪边框。输入可能是高分辨率的细笔画，因此不在源分辨率
    // 上增加固定 padding，避免 padding 比例随图片尺寸发生巨大变化。
    Tensor cropped;
    auto_crop(processed, cropped, 0, 0.02f);

    const auto& crop_shapes = cropped.shapes();
    const uint32_t crop_h = crop_shapes[crop_shapes.size() - 2];
    const uint32_t crop_w = crop_shapes[crop_shapes.size() - 1];
    if (crop_h == 0 || crop_w == 0) {
        output = Tensor({1, 28, 28});
        output.fill(0.0f);
        return;
    }

    // MNIST 字符通常位于约 20x20 的内容框内。保持宽高比缩放，给
    // 28x28 画布留下稳定边距。高分辨率下采样采用区域最大值，避免
    // 一到两个像素宽的手写笔画被双线性采样漏掉。
    constexpr uint32_t content_size = 20;
    uint32_t target_h = content_size;
    uint32_t target_w = content_size;
    if (crop_h >= crop_w) {
        target_w = std::max(1u, static_cast<uint32_t>(
            std::lround(static_cast<double>(content_size) * crop_w / crop_h)));
    } else {
        target_h = std::max(1u, static_cast<uint32_t>(
            std::lround(static_cast<double>(content_size) * crop_h / crop_w)));
    }

    Tensor scaled({1, target_h, target_w});
    if (crop_h > target_h || crop_w > target_w) {
        const float* source = cropped.raw_ptr();
        float* destination = scaled.raw_ptr();
        for (uint32_t y = 0; y < target_h; ++y) {
            const uint32_t y0 = y * crop_h / target_h;
            const uint32_t y1 = std::max(y0 + 1, (y + 1) * crop_h / target_h);
            for (uint32_t x = 0; x < target_w; ++x) {
                const uint32_t x0 = x * crop_w / target_w;
                const uint32_t x1 = std::max(x0 + 1, (x + 1) * crop_w / target_w);
                float value = 0.0f;
                for (uint32_t sy = y0; sy < std::min(y1, crop_h); ++sy) {
                    for (uint32_t sx = x0; sx < std::min(x1, crop_w); ++sx) {
                        value = std::max(value, source[sy * crop_w + sx]);
                    }
                }
                destination[y * target_w + x] = value;
            }
        }
    } else {
        resize(cropped, scaled, target_h, target_w);
    }

    Tensor canvas({1, 28, 28});
    canvas.fill(0.0f);
    const int offset_y = (28 - static_cast<int>(target_h)) / 2;
    const int offset_x = (28 - static_cast<int>(target_w)) / 2;
    for (uint32_t y = 0; y < target_h; ++y) {
        for (uint32_t x = 0; x < target_w; ++x) {
            canvas.raw_ptr()[(offset_y + y) * 28 + offset_x + x] =
                scaled.raw_ptr()[y * target_w + x];
        }
    }

    // 一次 3x3 最大值膨胀让细线条接近 MNIST 的笔画宽度。
    Tensor thickened({1, 28, 28});
    thickened.fill(0.0f);
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            float value = 0.0f;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int sy = y + dy;
                    const int sx = x + dx;
                    if (sy >= 0 && sy < 28 && sx >= 0 && sx < 28) {
                        value = std::max(value, canvas.raw_ptr()[sy * 28 + sx]);
                    }
                }
            }
            thickened.raw_ptr()[y * 28 + x] = value;
        }
    }

    // 按灰度质心平移到 MNIST 画布中心。
    double mass = 0.0;
    double center_y = 0.0;
    double center_x = 0.0;
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            const double value = thickened.raw_ptr()[y * 28 + x];
            mass += value;
            center_y += y * value;
            center_x += x * value;
        }
    }

    output = Tensor({1, 28, 28});
    output.fill(0.0f);
    const int shift_y = mass > 0.0 ? static_cast<int>(std::lround(13.5 - center_y / mass)) : 0;
    const int shift_x = mass > 0.0 ? static_cast<int>(std::lround(13.5 - center_x / mass)) : 0;
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            const int destination_y = y + shift_y;
            const int destination_x = x + shift_x;
            if (destination_y >= 0 && destination_y < 28 &&
                destination_x >= 0 && destination_x < 28) {
                output.raw_ptr()[destination_y * 28 + destination_x] =
                    thickened.raw_ptr()[y * 28 + x];
            }
        }
    }
}

//==========================================================================
// 私有辅助函数
//==========================================================================

bool Image::load_pgm(const std::string& filename,
                    std::vector<uint8_t>& buffer,
                    int& width, int& height)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    char magic[3];
    file.read(magic, 2);
    magic[2] = '\0';

    if (strcmp(magic, "P5") != 0) {
        file.close();
        return false;
    }

    file.get(); // skip whitespace

    // 跳过注释
    while (file.peek() == '#') {
        char line[256];
        file.getline(line, 256);
    }

    file >> width >> height;
    int maxval;
    file >> maxval;
    file.get(); // newline

    buffer.resize(width * height);
    file.read(reinterpret_cast<char*>(buffer.data()), width * height);
    file.close();

    return true;
}

void Image::resize_bilinear(const float* input,
                           float* output,
                           int in_h, int in_w,
                           int out_h, int out_w,
                           int channels)
{
    float x_ratio = static_cast<float>(in_w - 1) / out_w;
    float y_ratio = static_cast<float>(in_h - 1) / out_h;

    for (int c = 0; c < channels; ++c) {
        int c_offset = c * in_h * in_w;

        for (int y = 0; y < out_h; ++y) {
            for (int x = 0; x < out_w; ++x) {
                float px = x * x_ratio;
                float py = y * y_ratio;

                int x0 = static_cast<int>(px);
                int y0 = static_cast<int>(py);
                int x1 = std::min(x0 + 1, in_w - 1);
                int y1 = std::min(y0 + 1, in_h - 1);

                float dx = px - x0;
                float dy = py - y0;

                // 双线性插值
                float v00 = input[c_offset + y0 * in_w + x0];
                float v10 = input[c_offset + y0 * in_w + x1];
                float v01 = input[c_offset + y1 * in_w + x0];
                float v11 = input[c_offset + y1 * in_w + x1];

                float v0 = v00 * (1.0f - dx) + v10 * dx;
                float v1 = v01 * (1.0f - dx) + v11 * dx;
                float v = v0 * (1.0f - dy) + v1 * dy;

                int out_offset = c * out_h * out_w + y * out_w + x;
                output[out_offset] = v;
            }
        }
    }
}

} // namespace microflow
