# 模板特征缓存

## 数据结构

`tracker/template.py::TemplateCache` 保存初始模板特征、可选在线模板特征和当前 revision。`snapshot()` 返回不可变特征序列，推理只能读快照，不能在模型前向过程中修改缓存。

## 初始化

`initialize()` 检查 LocalView 和模板框，调用 HiTBackend.encodeTemplate，将结果作为永远保留的初始模板，并把 revision 建立到已知状态。RGB-D backend 对 RGB 和深度维护对应模板特征。

## 命令应用

每帧 Round 1 前，backend 按 TemplateCommand 执行：KEEP 只推进协议 revision；UPDATE 从指定 viewId 和 localBox 编码新特征，再原子替换在线部分。若会话不支持在线模板，UPDATE 必须按约定拒绝或保持安全回退，不能假装更新成功。

深度模板命令先准备特征，RGB 与深度两侧都成功后再提交，减少一边更新、一边失败导致的模态 revision 不一致。

## 为什么保留初始模板

在线模板能适应外观变化，但也可能漂移。推理特征序列始终保留初始模板，相当于长期身份锚点；在线模板提供短期适应。Controller 的稳定帧和重捕获冷却决定何时写入，详见 `Controller/templateAndTransaction.md`。

