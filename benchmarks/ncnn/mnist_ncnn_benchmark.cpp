#include <net.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

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
    const size_t index = static_cast<size_t>(fraction * static_cast<double>(values.size() - 1));
    return values[index];
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 5 || argc > 7) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.param> <model.bin> <images.idx> <labels.idx> [limit] [threads]\n";
        return 2;
    }

    try {
        std::ifstream images(argv[3], std::ios::binary);
        std::ifstream labels(argv[4], std::ios::binary);
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
        if (argc >= 6) count = std::min(count, static_cast<uint32_t>(parse_positive(argv[5], "limit")));
        const int threads = argc >= 7 ? parse_positive(argv[6], "threads") : 4;

        ncnn::Net net;
        net.opt.num_threads = threads;
        net.opt.lightmode = true;
        net.opt.use_packing_layout = true;
        net.opt.use_fp16_packed = false;
        net.opt.use_fp16_storage = false;
        net.opt.use_fp16_arithmetic = false;
        net.opt.use_bf16_storage = false;
        if (net.load_param(argv[1]) != 0 || net.load_model(argv[2]) != 0) {
            throw std::runtime_error("Cannot load ncnn model");
        }

        std::array<unsigned char, 28 * 28> pixels{};
        std::vector<double> inference_times;
        inference_times.reserve(count);
        uint32_t correct = 0;
        const auto wall_start = std::chrono::steady_clock::now();
        for (uint32_t sample = 0; sample < count; ++sample) {
            unsigned char label = 0;
            images.read(reinterpret_cast<char*>(pixels.data()), pixels.size());
            labels.read(reinterpret_cast<char*>(&label), 1);
            if (!images || !labels) throw std::runtime_error("Truncated MNIST IDX data");

            ncnn::Mat input(28, 28, 1);
            for (size_t i = 0; i < pixels.size(); ++i) {
                static_cast<float*>(input.data)[i] = static_cast<float>(pixels[i]) / 255.0f;
            }

            const auto start = std::chrono::steady_clock::now();
            ncnn::Extractor extractor = net.create_extractor();
            extractor.input("data", input);
            ncnn::Mat output;
            if (extractor.extract("prob", output) != 0) {
                throw std::runtime_error("ncnn inference failed");
            }
            const auto end = std::chrono::steady_clock::now();
            inference_times.push_back(
                std::chrono::duration<double, std::milli>(end - start).count());

            const float* scores = output;
            const int prediction = static_cast<int>(std::max_element(scores, scores + 10) - scores);
            if (prediction == label) ++correct;
        }
        const auto wall_end = std::chrono::steady_clock::now();

        const double wall_ms =
            std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
        const double average = std::accumulate(inference_times.begin(), inference_times.end(), 0.0) /
                               inference_times.size();
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Samples: " << count << '\n';
        std::cout << "Correct: " << correct << '\n';
        std::cout << "Accuracy: " << (100.0 * correct / count) << "%\n";
        std::cout << "Average inference: " << average << " ms\n";
        std::cout << "P50/P90/P99: " << percentile(inference_times, 0.50) << "/"
                  << percentile(inference_times, 0.90) << "/"
                  << percentile(inference_times, 0.99) << " ms\n";
        std::cout << "Wall throughput: " << (1000.0 * count / wall_ms) << " images/s\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ncnn benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
