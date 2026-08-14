# HiT 运行时

## 模型加载

`buildRuntime()` 默认构造 `PyTorchHiTSession`。会话要求 `model.backend: pytorch`、`model.variant: hit_small`，并在 CUDA 设备上加载 `models/hit_small.pth` 的 `net` 状态字典。官方源代码从 `third_party/HiT` 载入，配置文件为 `experiments/HiT/HiT_Small.yaml`。

模板区域按目标框扩展后裁剪并缩放到 `128 x 128`；搜索 RGB 缩放到 `256 x 256`。输入转换为 RGB 浮点张量，使用 ImageNet 均值和标准差归一化。

## 推理与置信度

HiT 输出归一化中心框和尺寸，适配器将其转换为局部像素框并限制在图像边界内。置信度由角点头热图的 softmax 熵和集中度计算，再作为模型分数和外观分数传入后端。非有限框或热图会触发模型错误。

FP16 配置在 CUDA 自动混合精度下运行；若边界框或两个角点热图出现非有限值，会清空热图并以 FP32 重新运行同一模型。

## RGB-only 与 RGB-D

| 线路 | 会话 | 输入 |
|---|---:|---|
| RGB-only | 1 | 局部 RGB |
| RGB-D | 2 | 局部 RGB；深度预处理生成的伪彩色 RGB |

RGB-D 的两个会话拥有独立模板特征和模型对象。深度摘要由 `DepthPreprocessor` 计算，并与 RGB 模型观测在 `FusionHead` 和控制器评估中组合。比赛入口通过配置校验拒绝深度线路。

## 生命周期

`TrackerBackendImpl.close()` 关闭 RGB 和深度会话。构造第二会话失败时，已创建的第一会话会关闭。会话关闭时移除热图钩子、释放模型引用并清理 CUDA 缓存。
