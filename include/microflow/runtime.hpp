#ifndef MICROFLOW_RUNTIME_HPP
#define MICROFLOW_RUNTIME_HPP

#include "microflow/tensor.hpp"
#include "microflow/conv.hpp"
#include "microflow/layers.hpp"
#include "microflow/gemm.hpp"

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <fstream>

namespace microflow {

//==========================================================================
// 模型文件格式定义 (.mflow v2)
//==========================================================================

/**
 * @brief 模型文件魔数
 */
constexpr uint32_t MFLOW_MAGIC = 0x4D464C57;  // "MFLW"

/**
 * @brief 层类型枚举
 */
enum class LayerType : uint32_t {
    kInput = 0,
    kConv2D = 1,
    kDepthwiseConv2D = 2,
    kPointwiseConv2D = 3,
    kBatchNorm = 4,
    kReLU = 5,
    kReLU6 = 6,
    kLeakyReLU = 7,
    kELU = 8,
    kGELU = 19,
    kMaxPool2D = 9,
    kAvgPool2D = 10,
    kGlobalAvgPool2D = 11,
    kAdaptiveAvgPool2D = 12,
    kLinear = 13,
    kFlatten = 14,
    kReshape = 15,
    kConcat = 16,
    kSoftmax = 17,
    kSigmoid = 18,
};

/**
 * @brief 数据类型枚举
 */
enum class DataType : uint32_t {
    kFloat32 = 0,
    kFloat16 = 1,
    kInt8 = 2,
    kUInt8 = 3,
};

/**
 * @brief 模型文件头
 */
struct ModelHeader {
    uint32_t magic;           // 魔数 "MFLW"
    uint32_t version;         // 版本号
    uint32_t num_layers;      // 层数
    uint32_t num_tensors;     // 张量数量
    uint32_t data_offset;     // 数据区偏移
    uint32_t data_size;       // 数据区大小
    char description[64];     // 模型描述
};

/**
 * @brief 层参数头
 */
struct LayerHeader {
    LayerType type;           // 层类型
    uint32_t name_offset;     // 层名字符串偏移
    uint32_t input_count;     // 输入数量
    uint32_t output_count;    // 输出数量
    uint32_t param_size;      // 参数大小
    uint32_t workspace_size;  // 工作空间大小
};

/**
 * @brief 张量描述
 */
struct TensorDesc {
    uint32_t ndim;            // 维度数
    uint32_t shapes[4];       // 形状 (最多4维)
    DataType dtype;           // 数据类型
    uint32_t size;            // 元素数量
    uint32_t offset;          // 在文件中的偏移
};

//==========================================================================
// 层基类
//==========================================================================

/**
 * @brief 层基类
 *
 * @设计考量:
 * - 虚析构函数确保正确派生类销毁
 * - 前向传播接口统一
 * - 支持工作空间管理
 */
class Layer {
public:
    virtual ~Layer() = default;

    /**
     * @brief 前向传播
     *
     * @param inputs 输入张量指针列表
     * @param outputs 输出张量指针列表
     * @param workspace 工作空间指针
     */
    virtual void forward(const std::vector<Tensor*>& inputs,
                       std::vector<Tensor*>& outputs,
                       float* workspace) = 0;

    /**
     * @brief 获取层名称
     */
    virtual std::string name() const = 0;

    /**
     * @brief 获取层类型
     */
    virtual LayerType type() const = 0;

    /**
     * @brief 获取所需工作空间大小
     */
    virtual size_t workspace_size() const = 0;

    /**
     * @brief 获取输出形状
     */
    virtual std::vector<uint32_t> output_shape(
        const std::vector<std::vector<uint32_t>>& input_shapes
    ) const = 0;
};

//==========================================================================
// 具体层实现
//==========================================================================

/**
 * @brief 输入层 (占位符)
 */
class InputLayer : public Layer {
public:
    InputLayer(const std::string& name, const std::vector<uint32_t>& shape);

    void forward(const std::vector<Tensor*>& inputs,
                std::vector<Tensor*>& outputs,
                float* workspace) override;

    std::string name() const override { return name_; }
    LayerType type() const override { return LayerType::kInput; }
    size_t workspace_size() const override { return 0; }

    std::vector<uint32_t> output_shape(
        const std::vector<std::vector<uint32_t>>& input_shapes
    ) const override;

public:
    std::string name_;
    std::vector<uint32_t> shape_;
};

/**
 * @brief 卷积层
 */
class Conv2DLayer : public Layer {
public:
    Conv2DLayer(const std::string& name,
               const Tensor& kernel,
               const Tensor& bias,
               const Conv2DParams& params);

    void forward(const std::vector<Tensor*>& inputs,
                std::vector<Tensor*>& outputs,
                float* workspace) override;

    std::string name() const override { return name_; }
    LayerType type() const override { return LayerType::kConv2D; }
    size_t workspace_size() const override;

    std::vector<uint32_t> output_shape(
        const std::vector<std::vector<uint32_t>>& input_shapes
    ) const override;

public:
    std::string name_;
    Tensor kernel_;
    Tensor winograd_kernel_;
    Tensor bias_;
    Conv2DParams params_;
    size_t workspace_size_;
    bool fuse_relu_ = false;
    bool fuse_maxpool_ = false;
};

/**
 * @brief Depthwise卷积层
 */
class DepthwiseConv2DLayer : public Layer {
public:
    DepthwiseConv2DLayer(const std::string& name,
                        const Tensor& kernel,
                        const Tensor& bias,
                        const Conv2DParams& params);

    void forward(const std::vector<Tensor*>& inputs,
                std::vector<Tensor*>& outputs,
                float* workspace) override;

    std::string name() const override { return name_; }
    LayerType type() const override { return LayerType::kDepthwiseConv2D; }
    size_t workspace_size() const override { return 0; }

    std::vector<uint32_t> output_shape(
        const std::vector<std::vector<uint32_t>>& input_shapes
    ) const override;

public:
    std::string name_;
    Tensor kernel_;
    Tensor bias_;
    Conv2DParams params_;
};

/**
 * @brief BatchNorm层
 */
class BatchNormLayer : public Layer {
public:
    BatchNormLayer(const std::string& name,
                  const Tensor& mean,
                  const Tensor& var,
                  const Tensor& gamma,
                  const Tensor& beta,
                  float eps = 1e-5f);

    void forward(const std::vector<Tensor*>& inputs,
                std::vector<Tensor*>& outputs,
                float* workspace) override;

    std::string name() const override { return name_; }
    LayerType type() const override { return LayerType::kBatchNorm; }
    size_t workspace_size() const override { return 0; }

    std::vector<uint32_t> output_shape(
        const std::vector<std::vector<uint32_t>>& input_shapes
    ) const override;

public:
    std::string name_;
    Tensor mean_;
    Tensor var_;
    Tensor gamma_;
    Tensor beta_;
    float eps_;
};

/**
 * @brief 激活层
 */
class ActivationLayer : public Layer {
public:
    ActivationLayer(const std::string& name, LayerType type);

    void forward(const std::vector<Tensor*>& inputs,
                std::vector<Tensor*>& outputs,
                float* workspace) override;

    std::string name() const override { return name_; }
    LayerType type() const override { return activation_type_; }
    size_t workspace_size() const override { return 0; }

    std::vector<uint32_t> output_shape(
        const std::vector<std::vector<uint32_t>>& input_shapes
    ) const override;

public:
    std::string name_;
    LayerType activation_type_;
};

/**
 * @brief 池化层
 */
class PoolingLayer : public Layer {
public:
    PoolingLayer(const std::string& name, LayerType type,
                int kernel_size, int stride, int padding = 0);

    void forward(const std::vector<Tensor*>& inputs,
                std::vector<Tensor*>& outputs,
                float* workspace) override;

    std::string name() const override { return name_; }
    LayerType type() const override { return pool_type_; }
    size_t workspace_size() const override { return 0; }

    std::vector<uint32_t> output_shape(
        const std::vector<std::vector<uint32_t>>& input_shapes
    ) const override;

public:
    std::string name_;
    LayerType pool_type_;
    int kernel_size_;
    int stride_;
    int padding_;
};

/**
 * @brief 全连接层
 */
class LinearLayer : public Layer {
public:
    LinearLayer(const std::string& name,
               const Tensor& weight,
               const Tensor& bias);

    void forward(const std::vector<Tensor*>& inputs,
                std::vector<Tensor*>& outputs,
                float* workspace) override;

    std::string name() const override { return name_; }
    LayerType type() const override { return LayerType::kLinear; }
    size_t workspace_size() const override { return 0; }

    std::vector<uint32_t> output_shape(
        const std::vector<std::vector<uint32_t>>& input_shapes
    ) const override;

public:
    std::string name_;
    Tensor weight_;
    Tensor packed_weight_;
    std::vector<int8_t> quantized_weight_;
    std::vector<float> quantized_weight_scales_;
    std::vector<int32_t> quantized_weight_sums_;
    std::vector<int8_t> quantized_input_;
    Tensor bias_;
    bool fuse_relu_ = false;
    bool dynamic_quantization_enabled_ = true;
};

/**
 * @brief Reshape层
 */
class ReshapeLayer : public Layer {
public:
    ReshapeLayer(const std::string& name,
                const std::vector<uint32_t>& shape);

    void forward(const std::vector<Tensor*>& inputs,
                std::vector<Tensor*>& outputs,
                float* workspace) override;

    std::string name() const override { return name_; }
    LayerType type() const override { return LayerType::kReshape; }
    size_t workspace_size() const override { return 0; }

    std::vector<uint32_t> output_shape(
        const std::vector<std::vector<uint32_t>>& input_shapes
    ) const override;

public:
    std::string name_;
    std::vector<uint32_t> shape_;
};

/**
 * @brief Flatten层
 */
class FlattenLayer : public Layer {
public:
    explicit FlattenLayer(const std::string& name);

    void forward(const std::vector<Tensor*>& inputs,
                std::vector<Tensor*>& outputs,
                float* workspace) override;

    std::string name() const override { return name_; }
    LayerType type() const override { return LayerType::kFlatten; }
    size_t workspace_size() const override { return 0; }

    std::vector<uint32_t> output_shape(
        const std::vector<std::vector<uint32_t>>& input_shapes
    ) const override;

public:
    std::string name_;
};

/**
 * @brief Softmax层
 */
class SoftmaxLayer : public Layer {
public:
    explicit SoftmaxLayer(const std::string& name, int axis = -1);

    void forward(const std::vector<Tensor*>& inputs,
                std::vector<Tensor*>& outputs,
                float* workspace) override;

    std::string name() const override { return name_; }
    LayerType type() const override { return LayerType::kSoftmax; }
    size_t workspace_size() const override { return 0; }

    std::vector<uint32_t> output_shape(
        const std::vector<std::vector<uint32_t>>& input_shapes
    ) const override;

public:
    std::string name_;
    int axis_;
};

//==========================================================================
// 模型类
//==========================================================================

/**
 * @brief 模型类
 *
 * @detail:
 * 管理整个神经网络模型,负责:
 * 1. 从文件加载模型
 * 2. 管理层之间的连接
 * 3. 分配和复用中间张量
 * 4. 执行推理
 */
class Model {
public:
    /**
     * @brief 构造函数
     */
    Model();
    Model(Model&&) noexcept = default;
    Model& operator=(Model&&) noexcept = default;
    // 同时也显式禁用拷贝，防止以后误用
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    /**
     * @brief 析构函数
     */
    ~Model();

    /**
     * @brief 从文件加载模型
     *
     * @param path 模型文件路径 (.mflow)
     * @return bool 成功返回true
     *
     * @文件格式:
     * - Header: 魔数、版本、层数等
     * - LayerHeaders: 每层的类型和参数
     * - TensorDescs: 权重张量描述
     * - StringTable: 层名称等字符串
     * - Data: 权重数据
     */
    bool load(const std::string& path);

    /**
     * @brief 保存模型到文件
     */
    bool save(const std::string& path);

    /**
     * @brief 添加层
     */
    void add_layer(std::unique_ptr<Layer> layer);

    /**
     * @brief 获取层
     */
    Layer* get_layer(const std::string& name);

    /**
     * @brief 推理
     *
     * @param input 输入张量
     * @param output 输出张量
     *
     * @优化点:
     * - 预分配中间张量
     * - 工作空间复用
     * - 层融合检测
     */
    void forward(const Tensor& input, Tensor& output);

    /**
     * @brief 批量推理
     */
    void forward_batch(const std::vector<Tensor>& inputs,
                      std::vector<Tensor>& outputs);

    /**
     * @brief 获取输入形状
     */
    std::vector<uint32_t> input_shape() const;

    /**
     * @brief 获取输出形状
     */
    std::vector<uint32_t> output_shape() const;

    /**
     * @brief 获取模型信息
     */
    struct Info {
        std::string name;
        std::string description;
        size_t num_layers;
        size_t num_parameters;
        size_t model_size;
    };
    Info get_info() const;

    /**
     * @brief 打印模型摘要
     */
    void summary() const;

public:
    /**
     * @brief 分配中间张量
     */
    void allocate_tensors();

    /**
     * @brief 计算所需工作空间
     */
    size_t compute_workspace_size();

    /**
     * @brief 层融合优化
     *
     * @detail:
     * 检测可以融合的层对:
     * - Conv + BN + ReLU
     * - Linear + ReLU
     * - 等
     */
    void fuse_layers();

    /** Enable or disable dynamic-int8 fully connected kernels. */
    void set_dynamic_quantization(bool enabled);

    std::vector<std::unique_ptr<Layer>> layers_;
    std::unordered_map<std::string, Layer*> layer_map_;

    // 中间张量缓存
    std::vector<Tensor> intermediate_tensors_;

    // 工作空间
    std::vector<float> workspace_;

    // 模型信息
    std::string name_;
    std::vector<uint32_t> input_shape_;
    std::vector<uint32_t> output_shape_;
    bool is_loaded_;
};

//==========================================================================
// 模型构建器
//==========================================================================

/**
 * @brief 模型构建器 (流式API)
 *
 * @detail:
 * 提供便捷的API构建模型
 *
 * @示例:
 * Model model = ModelBuilder("MyModel")
 *     .input({1, 28, 28})
 *     .conv2d("conv1", 8, 3, 1, 1)
 *     .relu()
 *     .max_pool(2, 2)
 *     .flatten()
 *     .linear("fc", 10)
 *     .softmax()
 *     .build();
 */
class ModelBuilder {
public:
    explicit ModelBuilder(const std::string& name);

    /**
     * @brief 设置输入层
     */
    ModelBuilder& input(const std::vector<uint32_t>& shape);

    /**
     * @brief 添加卷积层
     *
     * @param name 层名称
     * @param out_channels 输出通道数
     * @param kernel_size 卷积核大小
     * @param stride 步长
     * @param padding 填充
     * @param bias 是否使用偏置
     */
    ModelBuilder& conv2d(const std::string& name,
                        int out_channels,
                        int kernel_size = 3,
                        int stride = 1,
                        int padding = 0,
                        bool bias = true);

    /**
     * @brief 添加Depthwise卷积
     */
    ModelBuilder& depthwise_conv2d(const std::string& name,
                                  int kernel_size = 3,
                                  int stride = 1,
                                  int padding = 0,
                                  bool bias = true);

    /**
     * @brief 添加BatchNorm
     */
    ModelBuilder& batch_norm(const std::string& name);

    /**
     * @brief 添加ReLU激活
     */
    ModelBuilder& relu();

    /**
     * @brief 添加ReLU6激活
     */
    ModelBuilder& relu6();

    /**
     * @brief 添加GeLU激活
     */
    ModelBuilder& gelu();

    /**
     * @brief 添加LeakyReLU
     */
    ModelBuilder& leaky_relu(float alpha = 0.01f);

    /**
     * @brief 添加最大池化
     */
    ModelBuilder& max_pool(int kernel_size, int stride = 0, int padding = 0);

    /**
     * @brief 添加平均池化
     */
    ModelBuilder& avg_pool(int kernel_size, int stride = 0, int padding = 0);

    /**
     * @brief 添加全局平均池化
     */
    ModelBuilder& global_avg_pool();

    /**
     * @brief 添加全连接层
     */
    ModelBuilder& linear(const std::string& name,
                        int out_features,
                        bool bias = true);

    /**
     * @brief 展平
     */
    ModelBuilder& flatten();

    /**
     * @brief 重塑
     */
    ModelBuilder& reshape(const std::vector<uint32_t>& shape);

    /**
     * @brief Softmax
     */
    ModelBuilder& softmax(int axis = -1);

    /**
     * @brief 构建模型
     */
    Model build();

public:
    std::string name_;
    std::vector<std::unique_ptr<Layer>> layers_;
    std::vector<uint32_t> current_shape_;
};

//==========================================================================
// 推理引擎
//==========================================================================

/**
 * @brief 推理引擎
 *
 * @detail:
 * 高性能推理执行引擎,负责:
 * 1. 线程池管理
 * 2. 内存池管理
 * 3. 批处理调度
 * 4. 性能统计
 */
class InferenceEngine {
public:
    /**
     * @brief 配置参数
     */
    struct Config {
        int num_threads = -1;        // -1表示使用所有CPU核心
        size_t memory_pool_size = 16 * 1024 * 1024;  // 16MB
        bool enable_profiling = false;
    };

    explicit InferenceEngine(const Config& config);

    /**
     * @brief 加载模型
     */
    bool load_model(const std::string& path);

    /**
     * @brief 推理
     */
    Tensor infer(const Tensor& input);

    /**
     * @brief Infer into a caller-owned output tensor to avoid per-call allocation.
     */
    void infer_into(const Tensor& input, Tensor& output);

    /**
     * @brief 批量推理
     */
    std::vector<Tensor> infer_batch(const std::vector<Tensor>& inputs);

    /**
     * @brief 获取性能统计
     */
    struct Stats {
        double total_time_ms;
        double avg_time_ms;
        double min_time_ms;
        double max_time_ms;
        size_t num_inferences;
        double throughput;  // 推理/秒
    };
    Stats get_stats() const;

    /**
     * @brief 重置统计
     */
    void reset_stats();

    /**
     * @brief 获取中间层输出（用于可视化）
     *
     * @return 中间层张量列表，每个元素是一层的输出
     */
    std::vector<Tensor> get_intermediate_outputs() const;

public:
    Model model_;
    Config config_;

    // 性能统计
    Stats stats_;
    std::vector<double> inference_times_;
    mutable std::mutex inference_mutex_;
};

//==========================================================================
// 辅助函数
//==========================================================================

/**
 * @brief 创建预训练的MNIST模型
 */
Model create_mnist_model();

/**
 * @brief 创建MobileNet V2模型
 */
Model create_mobilenet_v2_model(int alpha = 1);

/**
 * @brief 创建简单的CNN模型
 */
Model create_simple_cnn_model(const std::vector<uint32_t>& input_shape,
                             int num_classes);

} // namespace microflow

#endif // MICROFLOW_RUNTIME_HPP
