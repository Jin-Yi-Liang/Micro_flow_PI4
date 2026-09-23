#include "microflow/conv.hpp"
#include "microflow/gemm.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

using namespace microflow;

namespace {

bool close_enough(const Tensor& actual, const Tensor& expected,
                  float absolute_tolerance, float relative_tolerance) {
    if (actual.shapes() != expected.shapes()) return false;
    for (uint32_t i = 0; i < actual.size(); ++i) {
        const float a = actual.raw_ptr()[i];
        const float e = expected.raw_ptr()[i];
        const float limit = absolute_tolerance + relative_tolerance * std::abs(e);
        if (std::abs(a - e) > limit) {
            std::cerr << "Mismatch at " << i << ": actual=" << a
                      << ", expected=" << e << ", limit=" << limit << '\n';
            return false;
        }
    }
    return true;
}

bool test_batch_one_gemm() {
    Tensor a({1, 3136});
    Tensor b({3136, 128});
    Tensor actual({1, 128});
    Tensor expected({1, 128});

    for (uint32_t i = 0; i < a.size(); ++i) {
        a.raw_ptr()[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.03125f;
    }
    for (uint32_t i = 0; i < b.size(); ++i) {
        b.raw_ptr()[i] = static_cast<float>(static_cast<int>(i % 23) - 11) * 0.015625f;
    }

    gemm(a, b, actual);
    gemm_naive(a, b, expected);
    return close_enough(actual, expected, 2e-4f, 2e-4f);
}

bool test_packed_gemv() {
    Tensor input({3136});
    Tensor packed({128, 3136});
    Tensor bias({128});
    Tensor actual({128});
    Tensor matrix_input({1, 3136}, input.raw_ptr());
    Tensor conventional = packed.transpose(0, 1);
    Tensor expected_matrix({1, 128});

    for (uint32_t i = 0; i < input.size(); ++i) {
        input.raw_ptr()[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.03125f;
    }
    for (uint32_t i = 0; i < packed.size(); ++i) {
        packed.raw_ptr()[i] = static_cast<float>(static_cast<int>(i % 23) - 11) * 0.015625f;
    }
    for (uint32_t i = 0; i < bias.size(); ++i) {
        bias.raw_ptr()[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.01f;
    }

    // packed was filled after conventional was created, so refresh it.
    conventional = packed.transpose(0, 1);
    gemm_naive(matrix_input, conventional, expected_matrix);
    for (uint32_t i = 0; i < bias.size(); ++i) {
        expected_matrix.raw_ptr()[i] =
            std::max(expected_matrix.raw_ptr()[i] + bias.raw_ptr()[i], 0.0f);
    }
    gemv_packed(input, packed, bias, actual, true);
    Tensor expected({128}, expected_matrix.raw_ptr());
    return close_enough(actual, expected, 2e-4f, 2e-4f);
}

bool test_conv3x3(int channels, int height, int width, int filters) {
    Tensor input({static_cast<uint32_t>(channels), static_cast<uint32_t>(height),
                  static_cast<uint32_t>(width)});
    Tensor kernel({static_cast<uint32_t>(filters), static_cast<uint32_t>(channels), 3, 3});
    Tensor actual({static_cast<uint32_t>(filters), static_cast<uint32_t>(height),
                   static_cast<uint32_t>(width)});
    Tensor expected(actual.shapes());

    for (uint32_t i = 0; i < input.size(); ++i) {
        input.raw_ptr()[i] = static_cast<float>(static_cast<int>(i % 29) - 14) * 0.02f;
    }
    for (uint32_t i = 0; i < kernel.size(); ++i) {
        kernel.raw_ptr()[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.025f;
    }

    const Conv2DParams params(3, 1, 1);
    conv2d_direct(input, kernel, expected, params);
    conv2d(input, kernel, Tensor(), actual, params);
    return close_enough(actual, expected, 3e-5f, 3e-5f);
}

bool test_winograd(int channels, int height, int width, int filters) {
    Tensor input({static_cast<uint32_t>(channels), static_cast<uint32_t>(height),
                  static_cast<uint32_t>(width)});
    Tensor kernel({static_cast<uint32_t>(filters), static_cast<uint32_t>(channels), 3, 3});
    Tensor transformed({static_cast<uint32_t>(filters), static_cast<uint32_t>(channels), 16});
    Tensor bias({static_cast<uint32_t>(filters)});
    Tensor actual({static_cast<uint32_t>(filters), static_cast<uint32_t>(height),
                   static_cast<uint32_t>(width)});
    Tensor expected(actual.shapes());

    for (uint32_t i = 0; i < input.size(); ++i) {
        input.raw_ptr()[i] = static_cast<float>(static_cast<int>(i % 29) - 14) * 0.02f;
    }
    for (uint32_t i = 0; i < kernel.size(); ++i) {
        kernel.raw_ptr()[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.025f;
    }
    for (uint32_t i = 0; i < bias.size(); ++i) {
        bias.raw_ptr()[i] = static_cast<float>(static_cast<int>(i % 5) - 2) * 0.03f;
    }

    const Conv2DParams params(3, 1, 1);
    conv2d_direct(input, kernel, expected, params);
    for (int filter = 0; filter < filters; ++filter) {
        for (int i = 0; i < height * width; ++i) {
            float& value = expected.raw_ptr()[filter * height * width + i];
            value = std::max(value + bias.raw_ptr()[filter], 0.0f);
        }
    }

    winograd_transform_kernel_3x3(kernel, transformed);
    const int tile_count = ((height + 1) / 2) * ((width + 1) / 2);
    std::vector<float> workspace(static_cast<size_t>(tile_count) * channels * 16);
    conv2d_winograd_f2x2(input, transformed, bias, actual, true, workspace.data());
    return close_enough(actual, expected, 2e-4f, 2e-4f);
}

} // namespace

int main() {
    int failures = 0;
    if (!test_batch_one_gemm()) {
        std::cerr << "FAIL: batch-one GEMM\n";
        ++failures;
    }
    if (!test_packed_gemv()) {
        std::cerr << "FAIL: packed GEMV\n";
        ++failures;
    }
    if (!test_conv3x3(1, 28, 28, 8)) {
        std::cerr << "FAIL: first MNIST convolution\n";
        ++failures;
    }
    if (!test_conv3x3(8, 14, 14, 16)) {
        std::cerr << "FAIL: multi-channel MNIST convolution\n";
        ++failures;
    }
    if (!test_winograd(1, 28, 28, 8) || !test_winograd(8, 14, 14, 16)) {
        std::cerr << "FAIL: Winograd F(2x2,3x3)\n";
        ++failures;
    }

    if (failures == 0) {
        std::cout << "All operator correctness tests passed.\n";
        return 0;
    }
    return 1;
}
