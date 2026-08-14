# 跟踪后端模块

跟踪后端负责局部视图上的模板管理、模型推理、深度证据和观测封装。

## HiT 接口

`HiTSession` 只包含模板编码、推理和关闭三个操作。`HiTBackend` 在边界处检查 RGB 数组为 `uint8 [H,W,3]`，检查模板特征和 `HiTPrediction` 类型，并将第三方异常转换为项目模型错误。

生产会话为 `PyTorchHiTSession`，只接受 `backend=pytorch` 和 `variant=hit_small`。它从官方 HiT 源码构建 HiT-Small，使用 CUDA 加载权重，并支持在线模板特征列表。

## RGB-only 与 RGB-D

RGB-only 通过一个 `HiTBackend` 处理所有局部 RGB 视图。RGB-D 的 `TrackerBackendImpl` 同时维护 RGB HiT 和深度 HiT：深度输入先由 `DepthPreprocessor` 转换为伪彩色 RGB，再由第二个 HiT 会话推理；`DepthEncoder` 将两个结果和局部深度摘要提供给融合头。

最终 `LocalObservation` 包含局部框、模型分数、外观分数、深度分数、融合分数、可选深度摘要和推理耗时。控制器只接收该标准观测，不依赖具体模型实现。

## 精度与资源

FP32 配置直接运行全精度前向。FP16 配置使用 CUDA 自动混合精度；检测到非有限输出后，在同一会话中以 FP32 重算。`close()` 释放前向钩子、模型引用和 CUDA 缓存。
