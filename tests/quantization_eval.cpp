#include "microflow/runtime.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

using namespace microflow;

namespace {
uint32_t read_be32(std::istream& stream) {
    std::array<unsigned char, 4> bytes{};
    stream.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    if (!stream) throw std::runtime_error("Unexpected end of IDX file");
    return (static_cast<uint32_t>(bytes[0]) << 24) |
           (static_cast<uint32_t>(bytes[1]) << 16) |
           (static_cast<uint32_t>(bytes[2]) << 8) |
           static_cast<uint32_t>(bytes[3]);
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.mflow> <images.idx> <labels.idx> <threads>\n";
        return 2;
    }
    try {
        std::ifstream images(argv[2], std::ios::binary);
        std::ifstream labels(argv[3], std::ios::binary);
        if (!images || !labels) throw std::runtime_error("Cannot open IDX files");
        const uint32_t image_magic = read_be32(images);
        const uint32_t count = read_be32(images);
        const uint32_t rows = read_be32(images);
        const uint32_t columns = read_be32(images);
        const uint32_t label_magic = read_be32(labels);
        const uint32_t label_count = read_be32(labels);
        if (image_magic != 2051 || label_magic != 2049 || rows != 28 || columns != 28 ||
            count != label_count) throw std::runtime_error("Invalid IDX headers");

        InferenceEngine::Config config;
        config.num_threads = std::stoi(argv[4]);
        InferenceEngine engine(config);
        if (!engine.load_model(argv[1])) throw std::runtime_error("Cannot load model");

        Tensor input({1, 28, 28});
        Tensor quantized({10});
        Tensor fp32({10});
        std::array<unsigned char, 784> pixels{};
        uint32_t differences = 0;
        std::cout << std::fixed << std::setprecision(6);
        for (uint32_t sample = 0; sample < count; ++sample) {
            unsigned char label = 0;
            images.read(reinterpret_cast<char*>(pixels.data()), pixels.size());
            labels.read(reinterpret_cast<char*>(&label), 1);
            for (size_t i = 0; i < pixels.size(); ++i) {
                input.raw_ptr()[i] = static_cast<float>(pixels[i]) / 255.0f;
            }
            engine.model_.set_dynamic_quantization(true);
            engine.infer_into(input, quantized);
            engine.model_.set_dynamic_quantization(false);
            engine.infer_into(input, fp32);
            const int quantized_prediction = static_cast<int>(
                std::max_element(quantized.raw_ptr(), quantized.raw_ptr() + 10) - quantized.raw_ptr());
            const int fp32_prediction = static_cast<int>(
                std::max_element(fp32.raw_ptr(), fp32.raw_ptr() + 10) - fp32.raw_ptr());
            if (quantized_prediction != fp32_prediction) {
                std::array<float, 10> sorted{};
                std::copy(quantized.raw_ptr(), quantized.raw_ptr() + 10, sorted.begin());
                std::sort(sorted.begin(), sorted.end(), std::greater<float>());
                std::cout << "sample=" << sample << " label=" << static_cast<int>(label)
                          << " int8=" << quantized_prediction << " fp32=" << fp32_prediction
                          << " confidence=" << sorted[0]
                          << " margin=" << sorted[0] - sorted[1] << '\n';
                ++differences;
            }
        }
        std::cout << "Prediction differences: " << differences << '/' << count << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Quantization evaluation failed: " << error.what() << '\n';
        return 1;
    }
}
