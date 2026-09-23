/**
 * @file benchmark.cpp
 * @brief MicroFlow性能基准测试
 *
 * @测试项目:
 * - 内存分配性能
 * - GEMM性能
 * - 卷积性能
 * - 整体模型推理性能
 */

#include "microflow/runtime.hpp"
#include "microflow/tensor.hpp"
#include "microflow/gemm.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <limits>
#include <vector>

using namespace microflow;

//==========================================================================
// 辅助函数
//==========================================================================

/**
 * @brief 获取当前时间 (毫秒)
 */
double get_time_ms() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double, std::milli>(duration).count();
}

/**
 * @brief 打印分隔线
 */
void print_separator(const std::string& title = "") {
    std::cout << "\n  ";
    for (int i = 0; i < 60; ++i) {
        std::cout << "=";
    }
    if (!title.empty()) {
        std::cout << "\n  " << title;
    }
    std::cout << "\n";
}

/**
 * @brief 格式化数字
 */
std::string format_number(double n) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << n;
    return oss.str();
}

//==========================================================================
// 内存分配基准测试
//==========================================================================

void benchmark_memory_allocation() {
    print_separator("Memory Allocation Benchmark");

    const int num_allocations = 10000;
    const size_t alloc_size = 1024;  // 1KB

    std::cout << "  Allocation Size: " << alloc_size << " bytes\n";
    std::cout << "  Number of Allocations: " << num_allocations << "\n\n";

    // 测试标准malloc
    {
        std::cout << "  Testing standard malloc...\n";
        double start = get_time_ms();

        std::vector<void*> ptrs;
        for (int i = 0; i < num_allocations; ++i) {
            ptrs.push_back(std::malloc(alloc_size));
        }

        double mid = get_time_ms();

        for (void* ptr : ptrs) {
            std::free(ptr);
        }

        double end = get_time_ms();

        std::cout << "    Alloc time: " << (mid - start) << " ms\n";
        std::cout << "    Free time:  " << (end - mid) << " ms\n";
        std::cout << "    Total:      " << (end - start) << " ms\n";
        std::cout << "    Avg alloc:  " << format_number((mid - start) / num_allocations * 1000) << " µs\n";
    }

    // 测试BumpPtrAllocator
    {
        std::cout << "\n  Testing BumpPtrAllocator...\n";
        double start = get_time_ms();

        BumpPtrAllocator allocator(16 * 1024 * 1024);  // 16MB初始池

        double alloc_start = get_time_ms();

        std::vector<void*> ptrs;
        for (int i = 0; i < num_allocations; ++i) {
            ptrs.push_back(allocator.allocate(alloc_size));
        }

        double alloc_end = get_time_ms();

        allocator.reset();

        double end = get_time_ms();

        std::cout << "    Alloc time:  " << (alloc_end - alloc_start) << " ms\n";
        std::cout << "    Reset time:  " << (end - alloc_end) << " ms\n";
        std::cout << "    Total:       " << (end - alloc_start) << " ms\n";
        std::cout << "    Avg alloc:   " << format_number((alloc_end - alloc_start) / num_allocations * 1000) << " µs\n";

        auto stats = allocator.get_stats();
        std::cout << "    Total allocated: " << (stats.total_memory / 1024) << " KB\n";
    }

    std::cout << "\n  Note: arena reset releases all temporary allocations in one operation.\n";
}

//==========================================================================
// GEMM基准测试
//==========================================================================

void benchmark_gemm() {
    print_separator("GEMM Benchmark");

    std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> test_sizes = {
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
    };

    std::cout << "  Testing matrix multiplication: C = A * B\n";
    std::cout << "  Iterations per size: 10\n\n";

    std::cout << "  ┌────────┬────────┬────────┬──────────┬──────────┬─────────┐\n";
    std::cout << "  │    M   │    N   │    K   │ Time (ms) │  GFLOPS  │ Speedup │\n";
    std::cout << "  ├────────┼────────┼────────┼──────────┼──────────┼─────────┤\n";

    for (auto [M, N, K] : test_sizes) {
        // 创建矩阵
        Tensor A({M, K});
        Tensor B({K, N});
        Tensor C({M, N});
        Tensor C_ref({M, N});

        // 填充随机数据
        A = Tensor::randn({M, K}, 0.0f, 1.0f);
        B = Tensor::randn({K, N}, 0.0f, 1.0f);

        // 测试naive实现
        double time_naive = 0.0f;
        {
            auto start = get_time_ms();
            gemm_naive(A, B, C_ref);
            auto end = get_time_ms();
            time_naive = end - start;
        }

        // 测试优化实现
        double time_opt = 0.0f;
        int iterations = 10;

        for (int i = 0; i < iterations; ++i) {
            C.fill(0.0f);
            auto start = get_time_ms();
            gemm(A, B, C);
            auto end = get_time_ms();
            time_opt += (end - start);
        }
        time_opt /= iterations;

        // 计算GFLOPS
        double gflops = (2.0 * M * N * K) / (time_opt / 1000.0) / 1e9;

        // 计算加速比
        double speedup = time_naive / time_opt;

        // 打印结果
        std::cout << "  │ " << std::setw(6) << M
                  << " │ " << std::setw(6) << N
                  << " │ " << std::setw(6) << K
                  << " │ " << std::setw(8) << std::fixed << std::setprecision(2) << time_opt
                  << " │ " << std::setw(8) << std::setprecision(2) << gflops
                  << " │ " << std::setw(7) << std::setprecision(1) << speedup
                  << "x │\n";

        // 验证正确性
        bool correct = verify_gemm(C, C_ref, 1e-3f);
        if (!correct) {
            std::cout << "  ⚠ Warning: Results don't match naive implementation!\n";
        }
    }

    std::cout << "  └────────┴────────┴────────┴──────────┴──────────┴─────────┘\n";

    std::cout << "\n  Values above are measured on the current host; no theoretical peak is assumed.\n";
}

//==========================================================================
// 卷积基准测试
//==========================================================================

void benchmark_conv() {
    print_separator("Convolution Benchmark");

    struct ConvTest {
        uint32_t C_in, H, W;
        uint32_t C_out;
        uint32_t K;
        uint32_t stride, pad;
    };

    std::vector<ConvTest> tests = {
        {1, 28, 28, 8, 3, 1, 1},    // MNIST first conv
        {8, 14, 14, 16, 3, 1, 1},   // MNIST second conv
        {3, 224, 224, 64, 7, 2, 3}, // ImageNet first conv
        {64, 56, 56, 128, 3, 1, 1}, // ImageNet mid conv
    };

    std::cout << "  Testing 2D convolution\n";
    std::cout << "  Iterations per size: 5\n\n";

    std::cout << "  ┌──────┬───────┬───────┬────┬────┬────┬──────────┬─────────┐\n";
    std::cout << "  │ Cin  │  H*W  │ Cout  │  K │  s │  p │ Time (ms) │  MFLOP  │\n";
    std::cout << "  ├──────┼───────┼───────┼────┼────┼────┼──────────┼─────────┤\n";

    for (auto test : tests) {
        Tensor input({test.C_in, test.H, test.W});
        Tensor kernel({test.C_out, test.C_in, test.K, test.K});
        Tensor output;

        input.fill(1.0f);
        kernel.fill(0.1f);

        Conv2DParams params(test.K, test.stride, test.pad);

        // 计算输出尺寸
        uint32_t H_out = (test.H + 2 * test.pad - test.K) / test.stride + 1;
        uint32_t W_out = (test.W + 2 * test.pad - test.K) / test.stride + 1;
        output = Tensor({test.C_out, H_out, W_out});

        // 计算FLOPs
        int64_t flops = int64_t(test.C_out) * test.C_in * test.K * test.K *
                        H_out * W_out * 2;

        // 运行测试
        int iterations = 5;
        double total_time = 0.0;

        for (int i = 0; i < iterations; ++i) {
            auto start = get_time_ms();
            conv2d(input, kernel, Tensor(), output, params);
            auto end = get_time_ms();
            total_time += (end - start);
        }

        double avg_time = total_time / iterations;
        double mflops = flops / (avg_time / 1000.0) / 1e6;

        std::cout << "  │ " << std::setw(4) << test.C_in
                  << " │ " << std::setw(5) << (test.H * test.W)
                  << " │ " << std::setw(5) << test.C_out
                  << " │ " << std::setw(2) << test.K
                  << " │ " << std::setw(2) << test.stride
                  << " │ " << std::setw(2) << test.pad
                  << " │ " << std::setw(8) << std::fixed << std::setprecision(2) << avg_time
                  << " │ " << std::setw(7) << std::setprecision(1) << mflops
                  << " │\n";
    }

    std::cout << "  └──────┴───────┴───────┴────┴────┴────┴──────────┴─────────┘\n";
}

//==========================================================================
// 综合推理基准测试
//==========================================================================

void benchmark_inference() {
    print_separator("End-to-End Inference Benchmark");

    std::cout << "  Simulating MNIST CNN model:\n";
    std::cout << "    Input: [1, 28, 28]\n";
    std::cout << "    Conv1: 8 filters, 3x3\n";
    std::cout << "    ReLU\n";
    std::cout << "    MaxPool: 2x2\n";
    std::cout << "    Flatten\n";
    std::cout << "    FC: 1568 -> 10\n\n";

    // 创建测试张量
    Tensor input({1, 28, 28});
    input = Tensor::randn({1, 28, 28}, 0.0f, 1.0f);

    Tensor conv1_weight({8, 1, 3, 3});
    conv1_weight.fill(0.1f);

    Tensor pool1_out({8, 14, 14});
    Tensor fc_weight({1568, 10});
    Tensor fc_bias({10});
    Tensor fc_output({1, 10});

    conv1_weight.fill(0.1f);
    fc_weight.fill(0.1f);
    fc_bias.fill(0.0f);

    Conv2DParams conv_params(3, 1, 1);

    int iterations = 100;
    std::cout << "  Running " << iterations << " iterations...\n\n";

    auto start = get_time_ms();
    double min_time = std::numeric_limits<double>::max();
    double max_time = 0.0;

    for (int i = 0; i < iterations; ++i) {
        const double iteration_start = get_time_ms();

        // Conv1 + ReLU
        Tensor conv1_out({8, 28, 28});
        conv2d(input, conv1_weight, Tensor(), conv1_out, conv_params);
        relu(conv1_out);

        // Pool
        max_pool2d(conv1_out, pool1_out, 2, 2);

        // Flatten
        Tensor fc_input({1, 1568});
        flatten(pool1_out, fc_input);

        // FC
        linear(fc_input, fc_weight, fc_bias, fc_output);

        const double iteration_time = get_time_ms() - iteration_start;
        min_time = std::min(min_time, iteration_time);
        max_time = std::max(max_time, iteration_time);
    }

    auto end = get_time_ms();

    double total_time = end - start;
    double avg_time = total_time / iterations;
    double throughput = 1000.0 / avg_time;

    std::cout << "  Results:\n";
    std::cout << "    Total time:   " << std::fixed << std::setprecision(2) << total_time << " ms\n";
    std::cout << "    Average time: " << avg_time << " ms\n";
    std::cout << "    Throughput:   " << std::setprecision(1) << throughput << " inferences/sec\n";
    std::cout << "    Min time:     " << min_time << " ms\n";
    std::cout << "    Max time:     " << max_time << " ms\n";
}

//==========================================================================
// 主函数
//==========================================================================

int main(int argc, char** argv) {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║        MicroFlow Performance Benchmark                ║\n";
    std::cout << "║        ROCK 4D / ARM64 Optimized                      ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n";

    // 系统信息
    std::cout << "\n  System Information:\n";
    std::cout << "    Architecture: " <<
#if defined(__aarch64__)
        "ARM64"
#elif defined(__x86_64__)
        "x86_64"
#else
        "other"
#endif
        << "\n";
    std::cout << "    Build Type: Release (O3)\n";
    std::cout << "    NEON: " <<
#ifdef MICROFLOW_HAS_NEON
        "Enabled"
#else
        "Disabled"
#endif
        << "\n";
    std::cout << "    OpenMP: " <<
#ifdef _OPENMP
        "Enabled (" << _OPENMP << ")"
#else
        "Disabled"
#endif
        << "\n";

    // 运行基准测试
    try {
        benchmark_memory_allocation();
        benchmark_gemm();
        benchmark_conv();
        benchmark_inference();

        print_separator();
        std::cout << "  All benchmarks completed successfully!\n\n";

    } catch (const std::exception& e) {
        std::cerr << "\n  Error during benchmark: " << e.what() << "\n\n";
        return 1;
    }

    return 0;
}
