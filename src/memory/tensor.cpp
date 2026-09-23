#include "microflow/tensor.hpp"
#include "microflow/gemm.hpp"
#include <stdexcept>
#include <random>
#include <iomanip>
#include <cmath>

namespace microflow {

//==========================================================================
// TensorView 实现
//==========================================================================

void TensorView::compute_strides() {
    if (shapes_.empty()) {
        size_ = 0;
        return;
    }

    const size_t ndim = shapes_.size();
    strides_.resize(ndim);

    // Row-Major: 最后一个维度步长为1
    uint32_t stride = 1;
    size_ = 1;

    for (int i = ndim - 1; i >= 0; --i) {
        strides_[i] = stride;
        stride *= shapes_[i];
        size_ *= shapes_[i];
    }
}

uint32_t TensorView::compute_offset(const std::vector<uint32_t>& indices) const {
    uint32_t offset = 0;
    for (size_t i = 0; i < indices.size() && i < strides_.size(); ++i) {
        offset += indices[i] * strides_[i];
    }
    return offset;
}

TensorView TensorView::slice(uint32_t dim, uint32_t index) const {
    if (dim >= shapes_.size()) {
        throw std::out_of_range("Dimension out of range");
    }
    if (index >= shapes_[dim]) {
        throw std::out_of_range("Slice index out of range");
    }

    std::vector<uint32_t> new_shapes = shapes_;
    new_shapes.erase(new_shapes.begin() + dim);

    // 计算切片后的数据指针偏移
    size_t byte_offset = index * strides_[dim] * sizeof(float);
    float* new_data = reinterpret_cast<float*>(
        reinterpret_cast<char*>(data_) + byte_offset
    );

    return TensorView(new_data, new_shapes, layout_);
}

//==========================================================================
// Tensor 实现
//==========================================================================

// 拷贝构造函数 - 深拷贝数据
Tensor::Tensor(const Tensor& other)
    : shapes_(other.shapes_)
    , layout_(other.layout_)
    , is_view_(false)  // 拷贝总是创建新的拥有数据的张量
{
    compute_strides();

    if (size_ > 0) {
        allocate_memory();
        // 深拷贝数据
        std::memcpy(data_.get(), other.data_.get(), size_ * sizeof(float));
    }
}

// 拷贝赋值运算符 - 深拷贝数据
Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        shapes_ = other.shapes_;
        layout_ = other.layout_;
        is_view_ = false;  // 赋值后拥有独立数据
        compute_strides();

        if (size_ > 0) {
            allocate_memory();
            std::memcpy(data_.get(), other.data_.get(), size_ * sizeof(float));
        }
    }
    return *this;
}

Tensor::Tensor(const std::vector<uint32_t>& shapes, DataLayout layout)
    : shapes_(shapes), layout_(layout), is_view_(false)
{
    compute_strides();
    allocate_memory();
    fill(0.0f);
}

Tensor::Tensor(const std::vector<uint32_t>& shapes, float fill_value, DataLayout layout)
    : shapes_(shapes), layout_(layout), is_view_(false)
{
    compute_strides();
    allocate_memory();
    fill(fill_value);
}

Tensor::Tensor(const std::vector<uint32_t>& shapes, float* external_ptr, DataLayout layout)
    : shapes_(shapes), layout_(layout), is_view_(true)
{
    compute_strides();

    // 使用no-op删除器 - 不释放外部内存
    data_ = std::shared_ptr<float[]>(external_ptr, [](float* p) {
        // Do nothing - 外部拥有内存
    });
}

Tensor::Tensor(const std::vector<uint32_t>& shapes, const float* data, DataLayout layout)
    : shapes_(shapes), layout_(layout), is_view_(false)
{
    compute_strides();
    allocate_memory();
    copy_from(data);
}

void Tensor::compute_strides() {
    if (shapes_.empty()) {
        size_ = 0;
        return;
    }

    const size_t ndim = shapes_.size();
    strides_.resize(ndim);

    // Row-Major布局
    uint32_t stride = 1;
    size_ = 1;

    for (int i = ndim - 1; i >= 0; --i) {
        strides_[i] = stride;
        stride *= shapes_[i];
        size_ *= shapes_[i];
    }
}

void Tensor::allocate_memory() {
    if (size_ == 0) {
        data_ = nullptr;
        return;
    }

    // 使用全局Arena分配器
    // 自动处理对齐和内存管理
    float* ptr = static_cast<float*>(
        get_global_allocator().allocate(
            size_ * sizeof(float),
            kDefaultAlignment  // 64字节对齐
        )
    );

    // 创建自定义删除器
    // 对于Arena分配的内存，不需要单独释放
    // 由Arena统一管理
    data_ = std::shared_ptr<float[]>(ptr, [](float* p) {
        // Arena模式: 无需释放
    });
}

void Tensor::fill(float value) {
    if (!data_ || size_ == 0) return;

    float* ptr = data_.get();

    // 小张量使用简单循环
    if (size_ < 32) {
        for (uint32_t i = 0; i < size_; ++i) {
            ptr[i] = value;
        }
        return;
    }

    // 大张量使用std::fill（可能使用SIMD）
    std::fill(ptr, ptr + size_, value);
}

void Tensor::copy_from(const float* src, size_t count) {
    if (!src || !data_) return;

    size_t n = (count == 0 || count > size_) ? size_ : count;

    // 使用memcpy（编译器优化为SIMD）
    std::memcpy(data_.get(), src, n * sizeof(float));
}

void Tensor::copy_to(float* dst) const {
    if (!dst || !data_) return;

    std::memcpy(dst, data_.get(), size_ * sizeof(float));
}

void Tensor::set_view_of(const Tensor& other, const std::vector<uint32_t>& new_shapes) {
    // 验证元素总数相同
    uint32_t new_size = std::accumulate(
        new_shapes.begin(), new_shapes.end(),
        1, std::multiplies<uint32_t>()
    );

    if (new_size != other.size_) {
        throw std::invalid_argument("set_view_of: size mismatch");
    }

    // 设置为视图模式 - 共享 other 的数据
    // 首先确保成员变量被正确初始化
    shapes_ = new_shapes;
    layout_ = other.layout_;
    size_ = new_size;
    data_ = other.data_;  // 共享数据指针
    is_view_ = true;
    compute_strides();
}

Tensor Tensor::reshape(const std::vector<uint32_t>& new_shapes) const {
    // 验证元素总数相同
    uint32_t new_size = std::accumulate(
        new_shapes.begin(), new_shapes.end(),
        1, std::multiplies<uint32_t>()
    );

    if (new_size != size_) {
        throw std::invalid_argument("Reshape: size mismatch");
    }

    // 创建共享数据的新张量
    // 使用explicit构造函数而不是默认构造函数，避免未初始化的成员变量
    Tensor result;
    result.shapes_ = new_shapes;
    result.layout_ = layout_;
    result.size_ = size_;
    result.strides_ = strides_;  // 先复制步长，确保有效值
    result.data_ = data_;  // 共享数据
    result.is_view_ = true;
    // 重新计算步长以适应新形状
    result.compute_strides();

    return result;
}

Tensor Tensor::transpose(uint32_t dim0, uint32_t dim1) const {
    if (dim0 >= shapes_.size() || dim1 >= shapes_.size()) {
        throw std::out_of_range("Transpose: dimension out of range");
    }

    // 2D张量特殊处理 - 需要创建真正的转置数据副本
    // 因为GEMM等函数不使用步长，只假设连续行主序布局
    if (shapes_.size() == 2) {
        uint32_t rows = shapes_[0];
        uint32_t cols = shapes_[1];

        Tensor result;
        result.shapes_ = {cols, rows};  // 交换后的形状
        result.layout_ = layout_;
        result.size_ = size_;
        result.is_view_ = false;  // 这是一个独立的张量
        result.compute_strides();
        result.allocate_memory();

        // 执行真正的转置
        const float* src = raw_ptr();
        float* dst = result.raw_ptr();

        for (uint32_t i = 0; i < rows; ++i) {
            for (uint32_t j = 0; j < cols; ++j) {
                dst[j * rows + i] = src[i * cols + j];
            }
        }

        return result;
    }

    // 高维张量创建连续数据副本。输出坐标交换回输入坐标后，使用
    // 原始 stride 计算源元素位置。
    Tensor result;
    result.shapes_ = shapes_;
    std::swap(result.shapes_[dim0], result.shapes_[dim1]);
    result.layout_ = layout_;
    result.compute_strides();
    result.allocate_memory();

    std::vector<uint32_t> coordinates(shapes_.size());
    for (uint32_t output_index = 0; output_index < result.size_; ++output_index) {
        uint32_t remainder = output_index;
        for (size_t axis = 0; axis < result.shapes_.size(); ++axis) {
            coordinates[axis] = remainder / result.strides_[axis];
            remainder %= result.strides_[axis];
        }
        std::swap(coordinates[dim0], coordinates[dim1]);

        uint32_t input_index = 0;
        for (size_t axis = 0; axis < coordinates.size(); ++axis) {
            input_index += coordinates[axis] * strides_[axis];
        }
        result.raw_ptr()[output_index] = raw_ptr()[input_index];
    }

    return result;
}

Tensor Tensor::expand_dims(uint32_t dim) const {
    if (dim > shapes_.size()) {
        throw std::out_of_range("expand_dims: dimension out of range");
    }

    Tensor result;
    result.shapes_ = shapes_;
    result.shapes_.insert(result.shapes_.begin() + dim, 1);
    result.layout_ = layout_;
    result.size_ = size_;
    result.data_ = data_;  // 共享数据
    result.is_view_ = true;
    result.compute_strides();

    return result;
}

Tensor Tensor::squeeze(uint32_t dim) const {
    if (dim >= shapes_.size()) {
        throw std::out_of_range("squeeze: dimension out of range");
    }

    if (shapes_[dim] != 1) {
        throw std::invalid_argument("squeeze: can only squeeze dimension of size 1");
    }

    Tensor result;
    result.shapes_ = shapes_;
    result.shapes_.erase(result.shapes_.begin() + dim);
    result.layout_ = layout_;
    result.size_ = size_;
    result.data_ = data_;  // 共享数据
    result.is_view_ = true;
    result.compute_strides();

    return result;
}

bool Tensor::is_contiguous() const {
    if (shapes_.size() <= 1) return true;

    // 检查步长是否连续
    uint32_t expected_stride = 1;
    for (int i = shapes_.size() - 1; i >= 0; --i) {
        if (strides_[i] != expected_stride) {
            return false;
        }
        expected_stride *= shapes_[i];
    }
    return true;
}

void Tensor::print_meta() const {
    std::cout << "Tensor Meta:\n";
    std::cout << "  Shape: [";
    for (size_t i = 0; i < shapes_.size(); ++i) {
        std::cout << shapes_[i];
        if (i < shapes_.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";

    std::cout << "  Size: " << size_ << " (" << (size_ * sizeof(float)) << " bytes)\n";
    std::cout << "  Layout: ";
    switch (layout_) {
        case DataLayout::kNCHW: std::cout << "NCHW"; break;
        case DataLayout::kNHWC: std::cout << "NHWC"; break;
        case DataLayout::kCHW: std::cout << "CHW"; break;
        case DataLayout::kHWC: std::cout << "HWC"; break;
        default: std::cout << "Unknown"; break;
    }
    std::cout << "\n";

    std::cout << "  Contiguous: " << (is_contiguous() ? "Yes" : "No") << "\n";
    std::cout << "  View: " << (is_view_ ? "Yes" : "No") << "\n";
    std::cout << "  Data ptr: " << static_cast<void*>(data_.get()) << "\n";
}

void Tensor::print_content(size_t max_elements) const {
    if (!data_ || size_ == 0) {
        std::cout << "Tensor is empty\n";
        return;
    }

    const float* ptr = data_.get();
    size_t n = std::min(size_, static_cast<uint32_t>(max_elements));

    std::cout << "Tensor Content (" << size_ << " elements):\n  [";
    for (size_t i = 0; i < n; ++i) {
        std::cout << ptr[i];
        if (i < n - 1) std::cout << ", ";
    }
    if (size_ > max_elements) {
        std::cout << ", ... (" << (size_ - max_elements) << " more)";
    }
    std::cout << "]\n";
}

//==========================================================================
// 静态工厂方法
//==========================================================================

Tensor Tensor::zeros(const std::vector<uint32_t>& shapes, DataLayout layout) {
    return Tensor(shapes, 0.0f, layout);
}

Tensor Tensor::ones(const std::vector<uint32_t>& shapes, DataLayout layout) {
    return Tensor(shapes, 1.0f, layout);
}

Tensor Tensor::randn(const std::vector<uint32_t>& shapes, float mean, float std, DataLayout layout) {
    Tensor result(shapes, layout);

    // 简单的Box-Muller变换生成正态分布
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    float* ptr = result.raw_ptr();
    for (uint32_t i = 0; i < result.size(); i += 2) {
        float u1 = dis(gen);
        float u2 = dis(gen);

        // 避免log(0)
        u1 = std::max(u1, 1e-6f);

        float r = std::sqrt(-2.0f * std::log(u1));
        float theta = 2.0f * M_PI * u2;

        ptr[i] = mean + std * r * std::cos(theta);
        if (i + 1 < result.size()) {
            ptr[i + 1] = mean + std * r * std::sin(theta);
        }
    }

    return result;
}

//==========================================================================
// 全局操作函数
//==========================================================================

Tensor add(const Tensor& a, const Tensor& b) {
    // 简化实现：假设形状相同
    if (a.shapes() != b.shapes()) {
        throw std::invalid_argument("add: shape mismatch");
    }

    Tensor result(a.shapes(), a.layout());
    const float* ptr_a = a.raw_ptr();
    const float* ptr_b = b.raw_ptr();
    float* ptr_result = result.raw_ptr();

    // 使用OpenMP并行化
    #pragma omp parallel for
    for (int64_t i = 0; i < static_cast<int64_t>(a.size()); ++i) {
        ptr_result[i] = ptr_a[i] + ptr_b[i];
    }

    return result;
}

Tensor mul(const Tensor& a, const Tensor& b) {
    if (a.shapes() != b.shapes()) {
        throw std::invalid_argument("mul: shape mismatch");
    }

    Tensor result(a.shapes(), a.layout());
    const float* ptr_a = a.raw_ptr();
    const float* ptr_b = b.raw_ptr();
    float* ptr_result = result.raw_ptr();

    #pragma omp parallel for
    for (int64_t i = 0; i < static_cast<int64_t>(a.size()); ++i) {
        ptr_result[i] = ptr_a[i] * ptr_b[i];
    }

    return result;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::invalid_argument("matmul: only 2D tensors supported");
    }
    if (a.shapes()[1] != b.shapes()[0]) {
        throw std::invalid_argument("matmul: inner dimensions must match");
    }

    Tensor result({a.shapes()[0], b.shapes()[1]}, a.layout());
    gemm(a, b, result);
    return result;
}

Tensor concat(const std::vector<Tensor>& tensors, uint32_t dim) {
    if (tensors.empty()) {
        throw std::invalid_argument("concat: no tensors provided");
    }

    if (dim >= tensors[0].ndim()) {
        throw std::out_of_range("concat: dimension out of range");
    }

    // 计算输出形状并验证除拼接维度以外的形状。
    std::vector<uint32_t> out_shapes = tensors[0].shapes();
    out_shapes[dim] = 0;
    for (const auto& t : tensors) {
        if (t.ndim() != tensors[0].ndim()) {
            throw std::invalid_argument("concat: rank mismatch");
        }
        for (uint32_t axis = 0; axis < t.ndim(); ++axis) {
            if (axis != dim && t.shapes()[axis] != tensors[0].shapes()[axis]) {
                throw std::invalid_argument("concat: shape mismatch");
            }
        }
        out_shapes[dim] += t.shapes()[dim];
    }

    Tensor result(out_shapes, tensors[0].layout());
    float* dst_ptr = result.raw_ptr();
    const size_t inner = std::accumulate(
        out_shapes.begin() + dim + 1, out_shapes.end(), size_t{1},
        std::multiplies<size_t>());
    const size_t outer = std::accumulate(
        out_shapes.begin(), out_shapes.begin() + dim, size_t{1},
        std::multiplies<size_t>());

    size_t axis_offset = 0;
    for (const auto& t : tensors) {
        const size_t chunk = static_cast<size_t>(t.shapes()[dim]) * inner;
        for (size_t outer_index = 0; outer_index < outer; ++outer_index) {
            const size_t destination = outer_index * out_shapes[dim] * inner + axis_offset;
            const size_t source = outer_index * chunk;
            std::memcpy(dst_ptr + destination, t.raw_ptr() + source,
                        chunk * sizeof(float));
        }
        axis_offset += chunk;
    }

    return result;
}

std::vector<Tensor> split(const Tensor& tensor, uint32_t parts, uint32_t dim) {
    if (parts == 0) {
        throw std::invalid_argument("split: parts must be positive");
    }
    if (dim >= tensor.ndim()) {
        throw std::out_of_range("split: dimension out of range");
    }
    if (tensor.shapes()[dim] % parts != 0) {
        throw std::invalid_argument("split: dimension not evenly divisible");
    }

    std::vector<Tensor> results;
    results.reserve(parts);
    std::vector<uint32_t> part_shape = tensor.shapes();
    part_shape[dim] /= parts;
    for (uint32_t part = 0; part < parts; ++part) {
        results.emplace_back(part_shape, tensor.layout());
    }

    const size_t inner = std::accumulate(
        tensor.shapes().begin() + dim + 1, tensor.shapes().end(), size_t{1},
        std::multiplies<size_t>());
    const size_t outer = std::accumulate(
        tensor.shapes().begin(), tensor.shapes().begin() + dim, size_t{1},
        std::multiplies<size_t>());
    const size_t chunk = static_cast<size_t>(part_shape[dim]) * inner;
    const size_t full_axis = static_cast<size_t>(tensor.shapes()[dim]) * inner;
    for (size_t outer_index = 0; outer_index < outer; ++outer_index) {
        for (uint32_t part = 0; part < parts; ++part) {
            const size_t source = outer_index * full_axis + part * chunk;
            const size_t destination = outer_index * chunk;
            std::memcpy(results[part].raw_ptr() + destination,
                        tensor.raw_ptr() + source, chunk * sizeof(float));
        }
    }

    return results;
}

} // namespace microflow
