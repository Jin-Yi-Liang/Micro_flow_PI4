#include "microflow/runtime.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace microflow;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main() {
    try {
        InferenceEngine::Config config;
        config.num_threads = 2;
        InferenceEngine engine(config);
        const std::string model_path =
            std::string(MICROFLOW_SOURCE_DIR) + "/models/mnist_improved.mflow";
        require(engine.load_model(model_path), "valid model failed to load");

        Tensor input({1, 28, 28});
        for (uint32_t i = 0; i < input.size(); ++i) {
            input.raw_ptr()[i] = static_cast<float>(i % 256) / 255.0f;
        }
        Tensor expected({10});
        engine.infer_into(input, expected);

        const std::vector<Tensor> inputs = {input, input, input};
        engine.reset_stats();
        const std::vector<Tensor> outputs = engine.infer_batch(inputs);
        require(outputs.size() == inputs.size(), "batch output count mismatch");
        for (const Tensor& output : outputs) {
            require(output.shapes() == std::vector<uint32_t>{10},
                    "batch output shape mismatch");
            for (uint32_t i = 0; i < output.size(); ++i) {
                require(std::abs(output.raw_ptr()[i] - expected.raw_ptr()[i]) < 1e-6f,
                        "batch inference is not deterministic");
            }
        }
        require(engine.get_stats().num_inferences == inputs.size(),
                "batch inference statistics mismatch");

        bool rejected_bad_output = false;
        try {
            Tensor wrong_output({9});
            engine.infer_into(input, wrong_output);
        } catch (const std::invalid_argument&) {
            rejected_bad_output = true;
        }
        require(rejected_bad_output, "runtime accepted an invalid output shape");

        const std::string bad_model = "/tmp/microflow_bad_model.mflow";
        {
            std::ofstream stream(bad_model, std::ios::binary | std::ios::trunc);
            const std::array<char, 8> bytes = {'B', 'A', 'D', 'M', 'O', 'D', 'E', 'L'};
            stream.write(bytes.data(), bytes.size());
        }
        Model invalid;
        require(!invalid.load(bad_model), "runtime accepted a truncated model");
        std::remove(bad_model.c_str());

        std::cout << "Runtime tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Runtime test failed: " << error.what() << '\n';
        return 1;
    }
}
