# HiT 运行算法

## 接口分层

`hit_backend.py::HiTSession` 只强制要求四项能力：在线模板能力声明、编码模板、单图推理、关闭资源。`inferBatch()` 是可选扩展；`HiTBackend` 优先调用它，不支持时按输入顺序回退到单图 `infer()`。当前生产 TrackerBackend 始终只传入第 0 帧 anchor 特征，不使用在线模板。外层统一检查 RGB 形状、模板非空、返回数量和分数范围，并把第三方异常翻译为项目 `ModelError`。

真实实现位于 `pytorch_hit_session.py::PyTorchHiTSession`。它加载 `src/instatarget/vendor/hit` 中的 HiT-Small 结构和 `model.weights` checkpoint，选择 CUDA 设备并按 `model.precision` 决定 fp32/fp16。

## 模板编码

初始化局部框先经过 HiT 的 template crop/resize 逻辑，变成固定尺寸张量；网络模板分支输出的特征被 TemplateCache 保存。模板框必须位于 LocalView 内，非法框在进入第三方模型前失败。

## 搜索推理

`TrackerBackendImpl.infer()` 将本轮 LocalView 按稳定 viewId 顺序一次交给 `HiTBackend.inferBatch()`。真实 PyTorch 会话把所有搜索图缩放到 256×256，堆叠为 `[B, 3, 256, 256]`，并把单份 `[1, C, H, W]` 模板特征扩展到 batch size B；每轮只执行一次模型 forward。

网络输出的 bbox 和两个 corner heatmap 按 batch 维拆回每张图。bbox 转为对应 LocalView 的局部像素并裁剪到边界；每张图的 heatmap 有限性和集中度独立生成置信度。返回结果数量必须等于输入数量，并保持输入顺序。fp16 任一 bbox 或 heatmap 非有限时，PyTorch 会话以 fp32 重算整个 batch。

当前自然 batch size 为 TRACKING 4+4、UNCERTAIN 4+4、LOST 10。两轮不能合并，因为第二轮 ViewSpec 依赖第一轮 Fusor 结果。自定义/测试 session 若没有 `inferBatch()` 仍可运行，但会退回逐图 forward，性能不能按生产 PyTorch 会话估算。

## 模型参数

- `model.backend`：生产路径应为 `pytorch`。
- `model.variant`：当前为 `hit_small`。
- `model.weights`：checkpoint 路径。
- `model.precision`：RGB-only 当前为 fp32，RGB-D 当前为 fp16。

精度优化需要同时检查框数值稳定性和校准分数分布；fp16 加速不能只看模型前向是否成功。

