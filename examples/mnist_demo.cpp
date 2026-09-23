/**
 * @file mnist_demo.cpp
 * @brief MNIST手写数字识别示例
 *
 * @使用方法:
 * ./mnist_demo <model_path> [input_image_path] [iterations] [threads]
 */

#include "microflow/runtime.hpp"
#include "microflow/tensor.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace microflow;

/**
 * @brief 加载MNIST图像
 *
 * @param path 图像文件路径
 * @param output 输出张量 [1, 28, 28]
 *
 * @note 支持 uint8 格式 (784 bytes) 和 float32 格式 (3136 bytes)
 */
bool load_mnist_image(const std::string& path, Tensor& output) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open image file: " << path << "\n";
        return false;
    }

    // 首先获取文件大小以确定格式
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    float* ptr = output.raw_ptr();

    if (file_size == 784) {
        // uint8 格式 - 传统 MNIST 格式
        std::vector<uint8_t> buffer(784);
        file.read(reinterpret_cast<char*>(buffer.data()), 784);
        if (!file) {
            std::cerr << "Error: Failed to read image data\n";
            return false;
        }
        // 转换为float并归一化到[0, 1]
        for (int i = 0; i < 784; ++i) {
            ptr[i] = static_cast<float>(buffer[i]) / 255.0f;
        }
    } else if (file_size == 3136) {
        // float32 格式 - 784 * 4 字节
        std::vector<float> buffer(784);
        file.read(reinterpret_cast<char*>(buffer.data()), 3136);
        if (!file) {
            std::cerr << "Error: Failed to read image data\n";
            return false;
        }
        // 直接复制（数据已经在 [0, 1] 范围或已经是正确的浮点值）
        std::memcpy(ptr, buffer.data(), 3136);
    } else {
        std::cerr << "Error: Invalid image file size: " << file_size
                  << " (expected 784 bytes for uint8 or 3136 bytes for float32)\n";
        return false;
    }

    return true;
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
    std::cout << "║     ARM64 / ROCK 4D Optimized             ║\n";
    std::cout << "╚════════════════════════════════════════════╝\n";
    std::cout << "\n";

    // 检查命令行参数
    if (argc < 2 || argc > 5) {
        std::cout << "Usage: " << argv[0]
                  << " <model_path> [image_path] [iterations] [threads]\n\n";
        std::cout << "Arguments:\n";
        std::cout << "  model_path  - Path to .mflow model file\n";
        std::cout << "  image_path  - Path to MNIST image file (optional)\n\n";
        std::cout << "  iterations  - Benchmark iterations (default: 100)\n";
        std::cout << "  threads     - Inference threads (default: 4)\n\n";
        std::cout << "Example:\n";
        std::cout << "  " << argv[0] << " model/mnist.mflow input/sample3.bin\n\n";
        return 1;
    }

    std::string model_path = argv[1];
    std::string image_path = (argc > 2) ? argv[2] : "";
    int iterations = 100;
    int threads = 4;
    try {
        if (argc > 3) iterations = std::stoi(argv[3]);
        if (argc > 4) threads = std::stoi(argv[4]);
    } catch (const std::exception&) {
        std::cerr << "Error: iterations and threads must be integers\n";
        return 2;
    }
    if (iterations < 1 || threads < 1 || threads > 64) {
        std::cerr << "Error: iterations must be positive and threads must be 1..64\n";
        return 2;
    }

    // 创建推理引擎
    std::cout << "Initializing inference engine...\n";
    InferenceEngine::Config config;
    config.num_threads = threads;
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
        if (!load_mnist_image(image_path, input)) {
            std::cerr << "\nError: Failed to load image!\n\n";
            return 1;
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
    Tensor output({10});
    engine.infer_into(input, output);

    // 打印结果
    print_prediction(output);

    // 性能测试
    run_inference_benchmark(engine, input, iterations);

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
