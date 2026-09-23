# MicroFlow

MicroFlow 是一个面向 ARM64 边缘设备的轻量级 C++ 神经网络推理引擎。本分支针对 Radxa ROCK 4D（RK3576，4×Cortex-A72 + 4×Cortex-A53）进行了端到端优化，当前提供模型加载、MNIST 文件/图片识别、完整测试集评估、性能统计和浏览器手写识别服务。

项目不依赖 Python 才能执行推理；运行时由 C++17、OpenMP 和项目内置的图像/HTTP 组件组成。Python 只用于训练导出、下载测试集以及生成竞品基准模型。

## 当前实测结果

测试日期：2026-09-23。设备为 ROCK 4D，Debian 12，Linux 6.1.84，固定运行在 Cortex-A72 大核 CPU 4–7，Release 构建，4 个 OpenMP 线程。数据为官方 MNIST test set 的全部 10,000 张图片。

| 实现 | 模型/精度策略 | 正确数 | 准确率 | 平均延迟 | 整体吞吐量 |
|---|---|---:|---:|---:|---:|
| MicroFlow 优化前基线 | FP32 | 9,937/10,000 | 99.37% | 约 4.52 ms | 约 221 images/s |
| MicroFlow 当前版本 | Winograd + 混合 INT8/FP32 | 9,937/10,000 | 99.37% | **0.505 ms** | **1,959.4 images/s** |
| ncnn 20260526 | FP32，禁用 FP16/BF16 | 9,937/10,000 | 99.37% | 0.591 ms | 1,671.3 images/s |

表中性能取连续三轮完整测试的中位一轮；MicroFlow 三轮平均延迟为 0.505/0.514/0.500 ms，P99 为 0.545～0.556 ms，冷机最佳轮为 0.489 ms。当前版本相对原始 ROCK 4D 基线约加速 **9.0 倍**，没有损失测试集准确率。在相同权重、相同输入、相同四个大核和相同预测结果下，相比 ncnn 参考实现平均延迟低约 **14.6%**，整体吞吐量高约 **17.2%**。最终数据应以目标板执行 `scripts/microflow-rock4d eval 10000` 的输出为准；温度、CPU 调频与后台负载会影响延迟。

竞品基准使用官方 ncnn tag `20260526`（commit `e54f7b1f88434e1d844ea0551b880a1cfb079ce1`），参考程序位于 `benchmarks/ncnn/`。它从同一个 `.mflow` 文件导出权重，不重新训练模型。

## 系统能力

- `.mflow` V2/V3 二进制模型加载与格式校验：检查 magic、版本、层类型、张量维度、数据类型、形状和文件截断。
- NCHW 张量、64 字节对齐内存、外部内存 View 和零拷贝 Flatten/Reshape。
- Conv2D、DepthwiseConv2D、PointwiseConv2D、BatchNorm、ReLU/ReLU6、池化、Linear、Softmax 等算子。
- MNIST LeNet 风格模型：`Conv(1→32) → ReLU → Pool → Conv(32→64) → ReLU → Pool → Flatten → Linear(3136→128) → ReLU → Linear(128→10) → Softmax`。
- `.bin`、PGM、PPM、PNG、JPEG、BMP 等图片输入与 MNIST 自动裁剪/缩放预处理。
- 命令行单图识别、10,000 张测试集评估、混合量化一致性检查、算子基准和 Web 手写板。
- HTTP `/health`、`/predict`、`/visualize` 接口，严格校验 784 个有限的 `[0,1]` 像素值，并支持 CORS。
- 可复用输出张量和串行化的引擎访问，避免长时间服务时持续增长的临时分配及请求数据竞争。

## ROCK 4D 专项优化

推理热路径包含以下优化：

1. 3×3、stride 1、padding 1 卷积采用 Winograd F(2×2, 3×3)，模型加载时预变换卷积核。
2. Winograd 输出阶段融合 bias、ReLU 和 2×2 MaxPool，减少中间张量写回和独立算子调度。
3. ARM NEON 向量化卷积、GEMV 和 INT8 点积；非适配形状仍保留通用正确性路径。
4. 大型第一全连接层使用分块动态 INT8；置信边界样本自动回退 FP32，以保持 99.37% 的完整测试集精度。
5. 小型第二全连接层保留 FP32，避免量化开销超过计算收益。
6. Flatten 使用 View，Linear 权重在加载阶段预打包，推理阶段复用工作区和输出内存。
7. 单次 OpenMP 并行区覆盖完整卷积，避免细粒度并行调度成本。
8. ROCK 4D 是异构八核；默认只绑定 CPU 4–7 的四个 Cortex-A72 大核，防止线程漂移到 Cortex-A53 小核后性能下降。

## 快速开始

### 依赖

Debian/Ubuntu：

```bash
sudo apt-get install build-essential cmake ninja-build python3
```

### 一键构建与测试

```bash
git clone https://github.com/Jin-Yi-Liang/Micro_flow_PI4.git
cd Micro_flow_PI4

./scripts/microflow-rock4d build
./scripts/microflow-rock4d test
```

脚本默认生成 `build-rock4d/`。也可以手动构建：

```bash
cmake -S . -B build-rock4d -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-rock4d -j"$(nproc)"
ctest --test-dir build-rock4d --output-on-failure
```

### 单张识别

仓库内置的测试输入标签为 7：

```bash
./scripts/microflow-rock4d demo image/test_input.bin 1000
```

也可识别常见图片；拍照、扫描或非 28×28 图片建议启用预处理：

```bash
taskset -c 4-7 build-rock4d/image_inference \
  models/mnist_improved.mflow image/image_7.png 4 --preprocess
```

### 完整 MNIST 测试

```bash
./scripts/microflow-rock4d eval 10000
```

若测试集不存在，脚本会下载并校验 MNIST IDX 文件。评估程序输出正确数、准确率、平均延迟、P50/P90/P99、端到端吞吐量和 10×10 混淆矩阵。

### Web 手写识别系统

```bash
./scripts/microflow-rock4d serve 8080
```

在同一网络的浏览器打开 `http://<ROCK-4D-IP>:8080`，即可在画布上书写数字并查看预测概率和中间层特征图。

健康检查：

```bash
curl http://127.0.0.1:8080/health
```

预测接口接收 JSON：

```json
{
  "pixels": [0.0, 0.0, 0.5, 1.0]
}
```

`pixels` 必须恰好包含 784 个数。成功响应示例：

```json
{
  "digit": 7,
  "confidence": 0.999998,
  "scores": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.999998, 0.0, 0.0]
}
```

## 一键命令

```text
scripts/microflow-rock4d build
scripts/microflow-rock4d test
scripts/microflow-rock4d eval [测试图片数量]
scripts/microflow-rock4d demo [图片路径] [重复次数]
scripts/microflow-rock4d serve [端口]
```

可通过环境变量覆盖默认值：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MICROFLOW_BUILD_DIR` | `build-rock4d` | 构建目录 |
| `MICROFLOW_MODEL` | `models/mnist_improved.mflow` | 模型文件 |
| `MICROFLOW_DATA_DIR` | `data/MNIST/raw` | IDX 数据目录 |
| `MICROFLOW_THREADS` | `4` | OpenMP 线程数 |
| `MICROFLOW_BIG_CORES` | `4-7` | 推理绑定的 CPU 列表 |

## 测试覆盖

`ctest` 当前覆盖：

- Tensor 的所有权、View、复制、通用转置、矩阵乘法、拼接与拆分；
- `.bin` 与常见图片加载/预处理，以及真实细线手写图片识别；
- GEMM/GEMV、打包权重、直接卷积和 Winograd 数值一致性；
- 模型加载、重复推理、批量推理和非法/截断模型拒绝；
- 内置 MNIST 样本的端到端预测必须为 7。

推荐在每次算子优化后执行：

```bash
./scripts/microflow-rock4d test
./scripts/microflow-rock4d eval 10000
```

单元测试用于发现实现回归；完整 10,000 张评估才是准确率验收标准，两者不能互相替代。

## 与 ncnn 做可重复对比

本仓库只包含参考程序，不捆绑 ncnn 源码。安装 ncnn 后执行：

```bash
python3 tools/export_ncnn_reference.py \
  models/mnist_improved.mflow benchmarks/ncnn

cmake -S benchmarks/ncnn -B benchmarks/ncnn/build \
  -Dncnn_DIR=/path/to/ncnn/lib/cmake/ncnn
cmake --build benchmarks/ncnn/build -j4

OMP_NUM_THREADS=4 OMP_PROC_BIND=true OMP_PLACES=cores taskset -c 4-7 \
  benchmarks/ncnn/build/mnist_ncnn_benchmark \
  benchmarks/ncnn/mnist_improved.param \
  benchmarks/ncnn/mnist_improved.bin \
  data/MNIST/raw/t10k-images-idx3-ubyte \
  data/MNIST/raw/t10k-labels-idx1-ubyte 10000 4
```

公平对比必须固定相同权重、输入、CPU 亲和性、线程数、精度开关和测试数量，并同时核对准确率，不能只比较延迟数字。

## C++ 嵌入接口

```cpp
#include "microflow/runtime.hpp"

microflow::InferenceEngine::Config config;
config.num_threads = 4;
config.enable_profiling = true;
microflow::InferenceEngine engine(config);

if (!engine.load_model("models/mnist_improved.mflow")) {
    return 1;
}

microflow::Tensor input({1, 28, 28});
microflow::Tensor output({10});
// 将归一化到 [0,1] 的 28×28 灰度像素写入 input。
engine.infer_into(input, output);
```

高频调用应优先使用 `infer_into`，让调用者复用输出张量。`infer` 是便捷接口，会返回新张量。

## 目录结构

```text
Micro_flow_PI4/
├── include/microflow/       公共 C++ 头文件
├── src/                     张量、算子、图像和运行时实现
├── models/                  可直接运行的 .mflow 模型
├── examples/                CLI、图片和 Web 示例
├── tests/                   单元、运行时、准确率与性能测试
├── tools/                   训练导出、数据下载和竞品模型转换
├── benchmarks/ncnn/         同权重 ncnn 参考基准
├── scripts/microflow-rock4d ROCK 4D 统一操作入口
└── CMakeLists.txt
```

## 常见问题

### 为什么不用 8 个 CPU 核心？

RK3576 的 Cortex-A72 与 Cortex-A53 性能不同。这个小模型的单次推理计算量有限，使用全部八核会增加调度和同步成本。板端实测四个 A72 大核明显更快，因此脚本默认 `taskset -c 4-7`。

### 为什么启用 INT8 后仍有 FP32？

MicroFlow 使用的是混合路径：第一全连接层的主路径为动态 INT8，而卷积、第二全连接层及置信边界样本保留或回退 FP32。这比“所有层一律量化”更适合当前小模型，并且能够保持完整测试集 99.37% 的结果。

### 为什么照片识别可能不如 MNIST 测试集？

MNIST 是居中、黑底白字的 28×28 灰度分布。真实照片会引入背景、透视、笔画粗细和极性差异。对普通图片使用 `--preprocess`；若面向特定摄像头部署，还应使用该摄像头采集的数据做校准或增广训练。

### 如何确认性能数字不是缓存或短样本偶然值？

运行完整 10,000 张测试并关注 P50/P90/P99；重复至少三次，保持 CPU 亲和性、线程数和散热条件一致。不要用只跑一次单图的 wall time 作为最终结论。

## License

项目许可证见仓库中的 `LICENSE`（若分发前尚未添加许可证文件，请先明确授权条款）。
