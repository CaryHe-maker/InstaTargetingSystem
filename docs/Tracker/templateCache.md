# 模板特征缓存

## 数据结构

`tracker/template.py::TemplateCache` 保存初始 anchor、兼容动态槽和当前 revision。`snapshot()` 返回不可变快照，但当前 TrackerBackend 只读取其中的 anchor 作为 HiT 模板输入。

## 初始化

`initialize()` 检查 LocalView 和模板框，调用 HiTBackend.encodeTemplate，将 RGB 特征作为永远保留的初始模板，并把 revision 建立到已知状态。

## 命令应用

当前 Controller 每轮只发送 KEEP，因此 backend 只推进协议 revision，不重新编码模板。UPDATE 命令及动态槽仍保留给兼容调用，但即使动态槽被写入，RGB 推理也只接收第 0 帧 anchor。

## 为什么固定初始模板

当前实现优先避免误检导致的模板漂移，因此整个序列只使用第 0 帧标准模板。代价是长期外观变化不会通过在线模板适应。

