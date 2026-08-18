# 模板策略与帧事务

## 模板命令

Controller 不直接编码模板，而是产生 `TemplateCommand`。当前生产策略始终发送 KEEP，HiT 的模板内容固定为第 0 帧初始化得到的 anchor。命令仍带 `expectedRevision`，Tracker 只接受严格递增 revision；revision 推进不代表模板内容发生变化。

## 更新条件

`template_policy.py::TemplatePolicy` 无论状态、稳定帧数或候选支持性如何都返回 KEEP。`UPDATE_RECENT`、`UPDATE_STABLE` 和相关配置字段仅为协议兼容保留，不进入当前生产路径。

## 模板缓存

实际特征存储在 `tracker/template.py::TemplateCache`。TrackerBackend 每次推理只把第 0 帧 RGB anchor 特征传给 HiT。即使外部兼容调用写入动态槽，动态特征也不会进入模型推理。

## 帧事务

`FrameTransaction` 保存起始状态、attempt 记录、剩余视图预算、第二轮搜索中心和 RecoveryMemory 副本。每个 AttemptRecord 保存该轮投影结果及 StateEvaluator 结果。TRACKING/UNCERTAIN 第一轮先由 Fusor 选出搜索中心；第二轮提交时读取第一轮记录，与第二轮观测组成事务候选池并再次统一调用 Fusor。中间 round 不更新 current target、运动历史或模板策略，最终 StateEvaluator 输出经过状态机后，Controller 一次性提交。

## 为什么模板命令只在 Round 1 应用

同一帧和跨帧推理都使用相同的第 0 帧 anchor，因此不同视图的模板输入保持一致。每个 round 仍发送 KEEP 并推进 expectedRevision，以保留事务乱序和重复消费检查。

revision 与特征内容不是同一概念。Tracker 每消费一个 attempt 都应用一次命令并将 backend/template revision 加一，所以 Round 1 与 Round 2 的 expectedRevision 不同且严格递增；Round 2 的 KEEP 不改变模板特征集合。

## 固定模板的取舍

固定 anchor 避免短暂误检污染身份模板，但不会自适应长期外观变化。评估时应分别观察长序列外观漂移和误跟踪恢复能力；若未来重新启用在线模板，需要同步恢复策略、backend 输入和对应回归测试。

