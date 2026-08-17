# 模板策略与帧事务

## 模板命令

Controller 不直接编码模板，而是产生 `TemplateCommand`：KEEP 保持当前模板，UPDATE 使用某个已确认视图的局部框添加在线模板。命令带 `expectedRevision`，Tracker 只接受严格递增 revision。

## 更新条件

`template_policy.py::TemplatePolicy` 只在 TRACKING 中考虑更新。状态必须连续稳定达到 `stableFramesBeforeUpdate`，当前聚合必须存在代表视图和局部框，并通过支持性判断。重捕获后的 `reacquireCooldownFrames` 内稳定计数被视为 0，避免把刚找回但仍不确定的外观写入模板。

## 模板缓存

实际特征存储在 `tracker/template.py::TemplateCache`。初始模板永远存在；在线模板仅在会话声明支持时加入。Controller 只持有下一帧命令，不持有模型特征。

## 帧事务

`FrameTransaction` 保存起始状态、attempt 记录、剩余视图预算和 RecoveryMemory 副本。每个 AttemptRecord 保存该轮的局部图投影结果；后续 round 会读取此前所有记录，与本轮结果组成累计候选池重新评估。中间 round 仍然不更新 current target、运动历史或模板策略。最终 StateEvaluator 输出经过状态机后，Controller 一次性提交。

## 为什么模板命令只在 Round 1 应用

同一帧的多个 round 必须使用同一模板状态，否则不同视图的分数不可比较。Controller 因此只在 attemptIndex=0 发送待处理模板命令，后续 round 强制 KEEP。模板更新的来源是上一帧已提交测量，不是当前尚未完成的候选。

## 优化风险

提高模板更新频率可能改善外观变化，也可能在短暂误检时造成不可逆漂移。优化时应记录模板 revision、来源 viewId、更新时状态、后续若干帧置信度和是否进入 LOST，而不是只比较平均分数。

