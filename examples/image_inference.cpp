#include "microflow/image.hpp"
#include "microflow/runtime.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace microflow;

int main(int argc, char** argv) {
    if (argc < 3 || argc > 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.mflow> <image> [threads] [--preprocess]\n";
        return 2;
    }

    try {
        int threads = 4;
        bool preprocess = false;
        for (int i = 3; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--preprocess") {
                preprocess = true;
            } else {
                threads = std::stoi(option);
                if (threads <= 0) throw std::runtime_error("threads must be positive");
            }
        }

        Tensor loaded;
        if (!Image::load(argv[2], loaded)) {
            throw std::runtime_error("cannot load input image");
        }

        Tensor input({1, 28, 28});
        const auto& shape = loaded.shapes();
        if (preprocess) {
            Image::preprocess_mnist(loaded, input);
        } else if (shape.size() == 3 && shape[0] == 1 && shape[1] == 28 && shape[2] == 28) {
            input = loaded;
        } else {
            Image::resize(loaded, input, 28, 28);
        }

        InferenceEngine::Config config;
        config.num_threads = threads;
        config.enable_profiling = true;
        InferenceEngine engine(config);
        if (!engine.load_model(argv[1])) throw std::runtime_error("cannot load model");

        Tensor output({10});
        engine.infer_into(input, output);
        const float* scores = output.raw_ptr();
        const int digit = static_cast<int>(
            std::max_element(scores, scores + 10) - scores);

        std::cout << "Digit: " << digit << '\n';
        std::cout << std::fixed << std::setprecision(6)
                  << "Confidence: " << scores[digit] << '\n';
        std::cout << "Scores:";
        for (int i = 0; i < 10; ++i) std::cout << ' ' << scores[i];
        std::cout << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Image inference failed: " << error.what() << '\n';
        return 1;
    }
}
