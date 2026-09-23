/**
 * @file test_tensor.cpp
 * @brief Tensor单元测试
 */

#include "microflow/tensor.hpp"
#include "microflow/allocator.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace microflow;

// 测试辅助宏
#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "FAIL: " << msg << "\n"; \
            return false; \
        } \
    } while(0)

#define TEST_BEGIN(name) \
    bool test_##name() { \
        std::cout << "  Testing " << #name << "... "; \
        bool passed = true;

#define TEST_END() \
        std::cout << (passed ? "PASS\n" : "FAIL\n"); \
        return passed; \
    }

//==========================================================================
// Tensor测试
//==========================================================================

TEST_BEGIN(tensor_creation) {
    // 测试基本创建
    Tensor t1({2, 3, 4});
    TEST_ASSERT(t1.size() == 24, "Size should be 24");
    TEST_ASSERT(t1.ndim() == 3, "ndim should be 3");
    TEST_ASSERT(t1.is_valid() == true, "Tensor should be valid");

    // 测试零初始化
    const float* ptr = t1.raw_ptr();
    for (uint32_t i = 0; i < t1.size(); ++i) {
        TEST_ASSERT(std::abs(ptr[i]) < 1e-6f, "Should be zero initialized");
    }

    // 测试填充值
    Tensor t2({10}, 1.5f);
    TEST_ASSERT(t2.size() == 10, "Size should be 10");
    ptr = t2.raw_ptr();
    for (int i = 0; i < 10; ++i) {
        TEST_ASSERT(std::abs(ptr[i] - 1.5f) < 1e-6f, "Should be 1.5");
    }
TEST_END()}

TEST_BEGIN(tensor_copy) {
    Tensor t1({4});
    t1.fill(2.5f);

    Tensor t2 = t1;  // 拷贝构造
    TEST_ASSERT(t2.size() == 4, "Copied tensor should have same size");

    const float* ptr1 = t1.raw_ptr();
    const float* ptr2 = t2.raw_ptr();
    TEST_ASSERT(ptr1 != ptr2, "Should have different data pointers");

    for (int i = 0; i < 4; ++i) {
        TEST_ASSERT(std::abs(ptr1[i] - ptr2[i]) < 1e-6f, "Values should match");
    }
TEST_END()}

TEST_BEGIN(tensor_reshape) {
    Tensor t1({2, 3, 4});
    t1.fill(1.0f);

    Tensor t2 = t1.reshape({6, 4});
    TEST_ASSERT(t2.size() == 24, "Reshaped tensor should have same size");
    TEST_ASSERT(t2.shapes()[0] == 6, "First dim should be 6");
    TEST_ASSERT(t2.shapes()[1] == 4, "Second dim should be 4");
TEST_END()}

TEST_BEGIN(tensor_operations) {
    Tensor t1({4});
    Tensor t2({4});

    t1.fill(2.0f);
    t2.fill(3.0f);

    // 加法
    Tensor t3 = add(t1, t2);
    const float* ptr = t3.raw_ptr();
    for (int i = 0; i < 4; ++i) {
        TEST_ASSERT(std::abs(ptr[i] - 5.0f) < 1e-6f, "2 + 3 should be 5");
    }

    // 乘法
    t3 = mul(t1, t2);
    ptr = t3.raw_ptr();
    for (int i = 0; i < 4; ++i) {
        TEST_ASSERT(std::abs(ptr[i] - 6.0f) < 1e-6f, "2 * 3 should be 6");
    }
TEST_END()}

TEST_BEGIN(tensor_zeros_ones) {
    Tensor t1 = Tensor::zeros({3, 4});
    TEST_ASSERT(t1.size() == 12, "Size should be 12");

    const float* ptr = t1.raw_ptr();
    for (int i = 0; i < 12; ++i) {
        TEST_ASSERT(std::abs(ptr[i]) < 1e-6f, "Should be zero");
    }

    Tensor t2 = Tensor::ones({3, 4});
    ptr = t2.raw_ptr();
    for (int i = 0; i < 12; ++i) {
        TEST_ASSERT(std::abs(ptr[i] - 1.0f) < 1e-6f, "Should be one");
    }
TEST_END()}

TEST_BEGIN(tensor_transformations) {
    Tensor input({2, 2, 3});
    for (uint32_t i = 0; i < input.size(); ++i) input.raw_ptr()[i] = static_cast<float>(i);

    Tensor transposed = input.transpose(0, 2);
    TEST_ASSERT(transposed.shapes() == std::vector<uint32_t>({3, 2, 2}),
                "3D transpose shape should match");
    // output[2,1,0] = input[0,1,2] = 5
    TEST_ASSERT(std::abs(transposed.raw_ptr()[10] - 5.0f) < 1e-6f,
                "3D transpose data should match");

    Tensor left({2, 2});
    Tensor right({2, 1});
    left.raw_ptr()[0] = 1; left.raw_ptr()[1] = 2;
    left.raw_ptr()[2] = 3; left.raw_ptr()[3] = 4;
    right.raw_ptr()[0] = 5; right.raw_ptr()[1] = 6;
    Tensor joined = concat({left, right}, 1);
    const float joined_expected[] = {1, 2, 5, 3, 4, 6};
    TEST_ASSERT(joined.shapes() == std::vector<uint32_t>({2, 3}),
                "Concat shape should match");
    for (uint32_t i = 0; i < joined.size(); ++i) {
        TEST_ASSERT(std::abs(joined.raw_ptr()[i] - joined_expected[i]) < 1e-6f,
                    "Concat data should match");
    }

    auto pieces = split(joined, 3, 1);
    TEST_ASSERT(pieces.size() == 3, "Split should return requested parts");
    TEST_ASSERT(pieces[0].shapes() == std::vector<uint32_t>({2, 1}),
                "Split shape should match");
    TEST_ASSERT(pieces[0].raw_ptr()[0] == 1 && pieces[0].raw_ptr()[1] == 3 &&
                pieces[2].raw_ptr()[0] == 5 && pieces[2].raw_ptr()[1] == 6,
                "Split data should match");

    Tensor a({2, 3});
    Tensor b({3, 2});
    for (uint32_t i = 0; i < a.size(); ++i) a.raw_ptr()[i] = static_cast<float>(i + 1);
    for (uint32_t i = 0; i < b.size(); ++i) b.raw_ptr()[i] = static_cast<float>(i + 1);
    Tensor product = matmul(a, b);
    const float product_expected[] = {22, 28, 49, 64};
    for (uint32_t i = 0; i < product.size(); ++i) {
        TEST_ASSERT(std::abs(product.raw_ptr()[i] - product_expected[i]) < 1e-5f,
                    "Matmul data should match");
    }
TEST_END()}

//==========================================================================
// 分配器测试
//==========================================================================

TEST_BEGIN(allocator_basic) {
    BumpPtrAllocator alloc(1024, 512);

    TEST_ASSERT(alloc.get_used_memory() == 0, "Initial used memory should be 0");

    // 分配一些内存
    void* p1 = alloc.allocate(100);
    TEST_ASSERT(p1 != nullptr, "Allocation should succeed");
    TEST_ASSERT(alloc.get_used_memory() >= 100, "Used memory should increase");

    void* p2 = alloc.allocate(200);
    TEST_ASSERT(p2 != nullptr, "Second allocation should succeed");

    // 重置
    alloc.reset();
    TEST_ASSERT(alloc.get_used_memory() == 0, "After reset, used memory should be 0");

    // 复用内存
    void* p3 = alloc.allocate(50);
    TEST_ASSERT(p3 != nullptr, "Allocation after reset should succeed");
TEST_END()}

TEST_BEGIN(allocator_alignment) {
    BumpPtrAllocator alloc;

    // 测试对齐
    void* p1 = alloc.allocate(100, 64);
    TEST_ASSERT((reinterpret_cast<size_t>(p1) % 64) == 0,
               "Should be 64-byte aligned");

    void* p2 = alloc.allocate(100, 256);
    TEST_ASSERT((reinterpret_cast<size_t>(p2) % 256) == 0,
               "Should be 256-byte aligned");
TEST_END()}

//==========================================================================
// 主函数
//==========================================================================

int main() {
    std::cout << "\n  MicroFlow Tensor Tests\n";
    std::cout << "  ========================\n\n";

    int passed = 0;
    int total = 0;

    // 运行测试
    #define RUN_TEST(test) \
        do { \
            total++; \
            if (test_##test()) passed++; \
        } while(0)

    RUN_TEST(tensor_creation);
    RUN_TEST(tensor_copy);
    RUN_TEST(tensor_reshape);
    RUN_TEST(tensor_operations);
    RUN_TEST(tensor_zeros_ones);
    RUN_TEST(tensor_transformations);
    RUN_TEST(allocator_basic);
    RUN_TEST(allocator_alignment);

    // 打印总结
    std::cout << "\n  ========================\n";
    std::cout << "  Results: " << passed << "/" << total << " tests passed\n";

    if (passed == total) {
        std::cout << "  All tests PASSED!\n\n";
        return 0;
    } else {
        std::cout << "  Some tests FAILED!\n\n";
        return 1;
    }
}
