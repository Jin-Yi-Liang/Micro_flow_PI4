/**
 * @file mnist_demo.cpp
 * @brief MNIST手写数字识别示例
 *
 * @使用方法:
 * ./mnist_demo <model_path> <input_image_path>
 */

#include "microflow/runtime.hpp"
#include "microflow/tensor.hpp"
#include "microflow/image.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace microflow;

/**
 * @brief 加载图像 (支持多种格式)
 *
 * @param path 图像文件路径
 * @param output 输出张量 [1, 28, 28]
 *
 * @note 支持 .bin (MNIST), .pgm (灰度图), .ppm (彩色图)
 */
bool load_mnist_image(const std::string& path, Tensor& output) {
    return Image::load(path, output);
}

/**
 * @brief 打印MNIST图像 (ASCII艺术)
 */
void print_mnist_image(const Tensor& image) {
    const float* ptr = image.raw_ptr();

    std::cout << "\n  MNIST Image (28x28):\n";
    std::cout << "  ";
    for (int i = 0; i < 28; ++i) {
        std::cout << "-";
    }
    std::cout << "\n";

    for (int h = 0; h < 28; ++h) {
        std::cout << "  |";
        for (int w = 0; w < 28; ++w) {
            float val = ptr[h * 28 + w];
            char ch = ' ';
            if (val > 0.9) ch = '@';
            else if (val > 0.7) ch = 'O';
            else if (val > 0.5) ch = 'o';
            else if (val > 0.3) ch = ':';
            else if (val > 0.1) ch = '.';
            std::cout << ch;
        }
        std::cout << "|\n";
    }

    std::cout << "  ";
    for (int i = 0; i < 28; ++i) {
        std::cout << "-";
    }
    std::cout << "\n\n";
}

/**
 * @brief 打印预测结果
 */
void print_prediction(const Tensor& output) {
    const float* ptr = output.raw_ptr();
    int num_classes = 10;

    std::cout << "  Prediction Scores:\n";
    std::cout << "  -----------------\n";

    // 找最大值
    float max_val = ptr[0];
    int predicted_digit = 0;

    for (int i = 0; i < num_classes; ++i) {
        std::cout << "  Digit " << i << ": "
                  << std::fixed << std::setprecision(6)
                  << ptr[i] << "  ";

        // 简单的条形图 (基于概率)
        int bar_len = static_cast<int>(ptr[i] * 50);
        for (int j = 0; j < bar_len; ++j) {
            std::cout << "▪";
        }
        std::cout << "\n";

        if (ptr[i] > max_val) {
            max_val = ptr[i];
            predicted_digit = i;
        }
    }

    std::cout << "\n  ========================================\n";
    std::cout << "  Predicted Digit: [" << predicted_digit << "]\n";
    std::cout << "  Confidence: " << std::fixed << std::setprecision(2)
              << (max_val * 100.0f) << "%\n";
    std::cout << "  ========================================\n\n";
}

/**
 * @brief 运行多次推理并统计性能
 */
void run_inference_benchmark(InferenceEngine& engine,
                            const Tensor& input,
                            int iterations)
{
    Tensor output({10});
    std::cout << "\n  Running " << iterations
              << " inference iterations...\n\n";

    // 预热
    for (int i = 0; i < 3; ++i) {
        engine.infer_into(input, output);
    }

    // 重置统计
    engine.reset_stats();

    // 正式测试
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        engine.infer_into(input, output);
    }

    auto end = std::chrono::high_resolution_clock::now();

    // 获取统计
    auto stats = engine.get_stats();

    // 打印结果
    std::cout << "  ========================================\n";
    std::cout << "  Performance Statistics:\n";
    std::cout << "  ========================================\n";
    std::cout << "  Total time:      " << std::fixed << std::setprecision(2)
              << stats.total_time_ms << " ms\n";
    std::cout << "  Average time:    " << stats.avg_time_ms << " ms\n";
    std::cout << "  Min time:        " << stats.min_time_ms << " ms\n";
    std::cout << "  Max time:        " << stats.max_time_ms << " ms\n";
    std::cout << "  Throughput:      " << std::fixed << std::setprecision(1)
              << stats.throughput << " inferences/sec\n";
    std::cout << "  ========================================\n\n";
}

/**
 * @brief 主函数
 */
int main(int argc, char** argv) {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════╗\n";
    std::cout << "║     MicroFlow MNIST Inference Demo        ║\n";
    std::cout << "║     ROCK 4D / ARM64 Optimized             ║\n";
    std::cout << "╚════════════════════════════════════════════╝\n";
    std::cout << "\n";

    // 检查命令行参数
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <model_path> [image_path] [options]\n\n";
        std::cout << "Arguments:\n";
        std::cout << "  model_path  - Path to .mflow model file\n";
        std::cout << "  image_path  - Path to image file (optional)\n\n";
        std::cout << "Options:\n";
        std::cout << "  --preprocess    Enable smart preprocessing (auto-crop, binarize)\n";
        std::cout << "                   Recommended for photos/scanned images\n\n";
        std::cout << "Example:\n";
        std::cout << "  " << argv[0] << " model/mnist.mflow photo.png --preprocess\n\n";
        return 1;
    }

    std::string model_path = argv[1];
    std::string image_path = (argc > 2) ? argv[2] : "";
    bool enable_preprocess = false;

    // 检查是否有 --preprocess 选项
    for (int i = 3; i < argc; ++i) {
        if (std::string(argv[i]) == "--preprocess") {
            enable_preprocess = true;
        }
    }

    // 创建推理引擎
    std::cout << "Initializing inference engine...\n";
    InferenceEngine::Config config;
    config.num_threads = 4;
    config.enable_profiling = true;
    InferenceEngine engine(config);

    // 加载模型
    std::cout << "Loading model from: " << model_path << "\n";
    if (!engine.load_model(model_path)) {
        std::cerr << "\nError: Failed to load model!\n\n";
        return 1;
    }
    std::cout << "Model loaded successfully!\n\n";

    // 准备输入
    Tensor input({1, 28, 28});

    if (!image_path.empty()) {
        // 从文件加载图像
        std::cout << "Loading image from: " << image_path << "\n";

        // 首先加载到临时张量（因为原始图像可能不是28x28）
        Tensor loaded;
        if (!load_mnist_image(image_path, loaded)) {
            std::cerr << "\nError: Failed to load image!\n\n";
            return 1;
        }

        // 显示原始图像信息
        auto shapes = loaded.shapes();
        std::cout << "  Original size: [" << shapes[0] << " channel, "
                  << shapes[1] << "x" << shapes[2] << " pixels]\n";

        // 选择处理方式
        if (enable_preprocess) {
            std::cout << "  Using smart preprocessing...\n";
            Image::preprocess_mnist(loaded, input);
        } else {
            // 简单缩放（原有逻辑）
            if (shapes[1] != 28 || shapes[2] != 28) {
                std::cout << "  Auto-resizing to 28x28...\n";
                Image::resize(loaded, input, 28, 28);
            } else {
                // 大小正确，直接使用
                input = loaded;
            }
        }

        // 调试：检查输入数据
        const float* ptr = input.raw_ptr();
        float min_val = ptr[0], max_val = ptr[0];
        int num_nonzero = 0;
        for (int i = 0; i < 784; ++i) {
            if (ptr[i] < min_val) min_val = ptr[i];
            if (ptr[i] > max_val) max_val = ptr[i];
            if (ptr[i] > 0.01f) num_nonzero++;
        }
        std::cout << "  Input stats: min=" << min_val << ", max=" << max_val
                  << ", nonzero=" << num_nonzero << "/784\n";

        print_mnist_image(input);
    } else {
        // 使用随机输入
        std::cout << "Using random input (no image file provided)\n";
        input = Tensor::randn({1, 28, 28}, 0.0f, 1.0f);
    }

    // 执行推理
    std::cout << "Running inference...\n";
    Tensor output = engine.infer(input);

    // 打印结果
    print_prediction(output);

    // 性能测试
    run_inference_benchmark(engine, input, 100);

    std::cout << "Demo completed successfully!\n\n";

    return 0;
}

/**
 * @mainpage MicroFlow Documentation
 *
 * @section intro Introduction
 * MicroFlow is a lightweight neural network inference engine optimized
 * for ROCK 4D Cortex-A72 big cores (ARM64 architecture).
 *
 * @section features Features
 * - ARM NEON optimized kernels
 * - Zero-copy tensor operations
 * - Layer fusion optimization
 * - Minimal memory footprint
 * - .mflow model format
 *
 * @section building Building
 * @code
 * mkdir build && cd build
 * cmake ..
 * make -j4
 * @endcode
 */
