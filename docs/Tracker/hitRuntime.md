# HiT 运行算法

## 接口分层

`hit_backend.py::HiTSession` 只要求四项能力：是否支持在线模板、编码模板、用模板特征推理搜索图、关闭资源。`HiTBackend` 在外层统一检查 RGB 形状、模板非空、分数范围，并把第三方异常翻译为项目 `ModelError`。

真实实现位于 `pytorch_hit_session.py::PyTorchHiTSession`。它加载 `src/instatarget/vendor/hit` 中的 HiT-Small 结构和 `model.weights` checkpoint，选择 CUDA 设备并按 `model.precision` 决定 fp32/fp16。

## 模板编码

初始化局部框先经过 HiT 的 template crop/resize 逻辑，变成固定尺寸张量；网络模板分支输出的特征被 TemplateCache 保存。模板框必须位于 LocalView 内，非法框在进入第三方模型前失败。

## 搜索推理

每个 LocalView 被缩放和标准化为 HiT 搜索张量，与缓存模板特征共同前向。网络输出归一化 bbox 和 heatmap；bbox 转为局部像素，随后裁剪到视图边界。heatmap 的有限性和峰值用于构造模型置信度，输出 `HiTPrediction`。

`TrackerBackendImpl.infer()` 对本轮视图按稳定 viewId 顺序调用 session。当前会话接口本质上逐视图调用；若未来改成真正 tensor batch，必须保证输出仍按输入顺序对应。

## 模型参数

- `model.backend`：生产路径应为 `pytorch`。
- `model.variant`：当前为 `hit_small`。
- `model.weights`：checkpoint 路径。
- `model.precision`：fp32 或 fp16。

精度优化需要同时检查框数值稳定性和校准分数分布；fp16 加速不能只看模型前向是否成功。

