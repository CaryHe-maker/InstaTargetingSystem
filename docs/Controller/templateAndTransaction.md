# 模板策略与帧事务

## 模板命令

Controller 不直接编码模板，而是产生 `TemplateCommand`：KEEP 保持当前模板，UPDATE 使用某个已确认视图的局部框添加在线模板。命令带 `expectedRevision`，Tracker 只接受严格递增 revision。

## 更新条件

`template_policy.py::TemplatePolicy` 只在 TRACKING 中考虑更新。状态必须连续稳定达到 `stableFramesBeforeUpdate`，当前聚合必须存在代表视图和局部框，并通过支持性判断。重捕获后的 `reacquireCooldownFrames` 内稳定计数被视为 0，避免把刚找回但仍不确定的外观写入模板。

## 模板缓存

实际特征存储在 `tracker/template.py::TemplateCache`。初始模板永远存在；在线模板仅在会话声明支持时加入。Controller 只持有下一帧命令，不持有模型特征。

## 帧事务

`FrameTransaction` 保存起始状态、attempt 记录、剩余视图预算、第二轮搜索中心和 RecoveryMemory 副本。每个 AttemptRecord 保存该轮投影结果及 StateEvaluator 结果。TRACKING/UNCERTAIN 第一轮先由 Fusor 选出搜索中心；第二轮提交时读取第一轮记录，与第二轮观测组成事务候选池并再次统一调用 Fusor。中间 round 不更新 current target、运动历史或模板策略，最终 StateEvaluator 输出经过状态机后，Controller 一次性提交。

## 为什么模板命令只在 Round 1 应用

同一帧的多个 round 必须使用相同的模板特征内容，否则不同视图的分数不可比较。Controller 因此只在 attemptIndex=0 发送待处理模板命令，后续 round 强制 KEEP；模板更新来源是上一帧已提交测量，不是当前尚未完成的候选。

revision 与特征内容不是同一概念。Tracker 每消费一个 attempt 都应用一次命令并将 backend/template revision 加一，所以 Round 1 与 Round 2 的 expectedRevision 不同且严格递增；Round 2 的 KEEP 不改变模板特征集合。

## 优化风险

提高模板更新频率可能改善外观变化，也可能在短暂误检时造成不可逆漂移。优化时应记录模板 revision、来源 viewId、更新时状态、后续若干帧置信度和是否进入 LOST，而不是只比较平均分数。

