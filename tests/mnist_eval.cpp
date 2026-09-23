#include "microflow/runtime.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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

int parse_positive(const char* value, const char* name) {
    const int parsed = std::stoi(value);
    if (parsed <= 0) throw std::runtime_error(std::string(name) + " must be positive");
    return parsed;
}

double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>(fraction * (values.size() - 1));
    return values[index];
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4 || argc > 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.mflow> <images.idx> <labels.idx> [limit] [threads]\n";
        return 2;
    }

    try {
        std::ifstream images(argv[2], std::ios::binary);
        std::ifstream labels(argv[3], std::ios::binary);
        if (!images || !labels) throw std::runtime_error("Cannot open MNIST IDX files");

        const uint32_t image_magic = read_be32(images);
        const uint32_t image_count = read_be32(images);
        const uint32_t rows = read_be32(images);
        const uint32_t columns = read_be32(images);
        const uint32_t label_magic = read_be32(labels);
        const uint32_t label_count = read_be32(labels);
        if (image_magic != 2051 || label_magic != 2049 || rows != 28 || columns != 28 ||
            image_count != label_count) {
            throw std::runtime_error("Invalid or mismatched MNIST IDX headers");
        }

        uint32_t count = image_count;
        if (argc >= 5) count = std::min(count, static_cast<uint32_t>(parse_positive(argv[4], "limit")));
        const int threads = argc >= 6 ? parse_positive(argv[5], "threads") : 4;

        InferenceEngine::Config config;
        config.num_threads = threads;
        config.enable_profiling = true;
        InferenceEngine engine(config);
        if (!engine.load_model(argv[1])) throw std::runtime_error("Cannot load model");

        Tensor input({1, 28, 28});
        Tensor output({10});
        std::array<unsigned char, 28 * 28> pixels{};
        std::array<std::array<uint32_t, 10>, 10> confusion{};

        uint32_t correct = 0;
        engine.reset_stats();
        const auto wall_start = std::chrono::steady_clock::now();
        for (uint32_t sample = 0; sample < count; ++sample) {
            unsigned char label = 0;
            images.read(reinterpret_cast<char*>(pixels.data()), pixels.size());
            labels.read(reinterpret_cast<char*>(&label), 1);
            if (!images || !labels) throw std::runtime_error("Truncated MNIST IDX data");

            for (size_t i = 0; i < pixels.size(); ++i) {
                input.raw_ptr()[i] = static_cast<float>(pixels[i]) / 255.0f;
            }
            engine.infer_into(input, output);
            const int prediction = static_cast<int>(
                std::max_element(output.raw_ptr(), output.raw_ptr() + 10) - output.raw_ptr());
            ++confusion[label][prediction];
            if (prediction == label) ++correct;
        }
        const auto wall_end = std::chrono::steady_clock::now();

        const auto stats = engine.get_stats();
        const double wall_ms =
            std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
        const double accuracy = 100.0 * static_cast<double>(correct) / count;

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Samples: " << count << '\n';
        std::cout << "Correct: " << correct << '\n';
        std::cout << "Accuracy: " << accuracy << "%\n";
        std::cout << "Average inference: " << stats.avg_time_ms << " ms\n";
        std::cout << "P50/P90/P99: "
                  << percentile(engine.inference_times_, 0.50) << "/"
                  << percentile(engine.inference_times_, 0.90) << "/"
                  << percentile(engine.inference_times_, 0.99) << " ms\n";
        std::cout << "Wall throughput: " << (1000.0 * count / wall_ms) << " images/s\n";
        std::cout << "Confusion matrix (rows=true, columns=predicted):\n";
        for (const auto& row : confusion) {
            for (uint32_t value : row) std::cout << std::setw(5) << value;
            std::cout << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MNIST evaluation failed: " << error.what() << '\n';
        return 1;
    }
}
