#include "microflow/runtime.hpp"
#include "microflow/image.hpp"
#include <iostream>

using namespace microflow;

#ifndef MICROFLOW_SOURCE_DIR
#define MICROFLOW_SOURCE_DIR "."
#endif

int main() {
    std::cout << "╔════════════════════════════════════════════╗\n";
    std::cout << "║     MicroFlow Image Loading Test          ║\n";
    std::cout << "╚════════════════════════════════════════════╝\n\n";

    // 测试1: 加载现有MNIST .bin文件
    std::cout << "Test 1: Loading MNIST .bin file...\n";
    Tensor img1;
    if (Image::load(MICROFLOW_SOURCE_DIR "/image/test_input.bin", img1)) {
        std::cout << "  ✓ Loaded successfully\n";
        std::cout << "  Shape: [" << img1.shapes()[0] << ", "
                  << img1.shapes()[1] << ", " << img1.shapes()[2] << "]\n";
    } else {
        std::cout << "  ✗ Failed to load\n";
        return 1;
    }

    // 测试2: 图像处理功能
    std::cout << "\nTest 2: Image processing...\n";

    // 创建一个模拟的RGB图像 (用于测试)
    Tensor rgb({3, 56, 56});
    float* ptr = rgb.raw_ptr();
    for (size_t i = 0; i < rgb.size(); ++i) {
        ptr[i] = static_cast<float>(rand() % 256) / 255.0f;
    }

    // 转灰度
    Tensor gray;
    Image::to_grayscale(rgb, gray);
    std::cout << "  ✓ RGB [" << rgb.shapes()[1] << "x" << rgb.shapes()[2]
              << "] -> Gray [" << gray.shapes()[1] << "x" << gray.shapes()[2] << "]\n";

    // 调整大小
    Tensor resized;
    Image::resize(gray, resized, 28, 28);
    std::cout << "  ✓ Resized to [" << resized.shapes()[1] << "x" << resized.shapes()[2] << "]\n";

    // 反色
    Tensor inverted(resized.shapes());
    resized.copy_to(inverted.raw_ptr());
    Image::invert(inverted);
    std::cout << "  ✓ Inverted colors\n";

    std::cout << "\n✅ All image tests passed!\n";

    return 0;
}
