# 状态机算法

## 两层状态

跨帧状态为 `TRACKING`、`UNCERTAIN`、`RECOVERING`、`LOST`；Round 1/2/3 只是当前帧的尝试序号，不是持久状态。`TrackStateMachine.transition()` 是纯约简函数，持久计数由 DepthAwareTrackController 保存。

## 证据分类

StateEvaluator 最终只向状态机提供三类核心测量证据：可靠融合、可靠单框、弱/缺失。可靠融合表示两个不同视域在 ERP 上高度重合，并且融合与两个来源分数都达标；可靠单框只在允许单框输出的轮次成立。

## 转移规则

| 当前状态 | 可靠融合 | 可靠单框 | 弱或缺失 |
|---|---|---|---|
| TRACKING | 保持 TRACKING，接受测量 | 保持 TRACKING，接受测量 | 进入 UNCERTAIN，不写测量历史 |
| UNCERTAIN | 回 TRACKING | 回 TRACKING | 未耗尽 `uncertainPatience` 时保持，否则进入 LOST |
| LOST | 直接回 TRACKING并重置运动历史 | 进入 RECOVERING 等待确认 | 保持 LOST |
| RECOVERING | 直接回 TRACKING | 连续达到 `recoverConfirmFrames` 后回 TRACKING | 回 LOST |

融合重捕获可以直接恢复，是因为两个独立视域提供了空间一致性；单框重捕获需要跨帧确认，避免全景搜索中的偶然高分误检。

## 提交副作用

只有 `acceptMeasurement=True` 才更新 current bbox/BFoV 和运动样本。LOST/RECOVERING 成功重捕获时清空旧速度假设并开启模板冷却。未接受帧仍可输出运动预测或弱观测框，但 `valid=False`，不能污染下一次拟合。

## 主要参数

`uncertainPatience` 控制弱跟踪能持续多久；`recoverConfirmFrames` 控制单框重捕获确认；`reacquireCooldownFrames` 防止刚找回就立刻更新模板。修改这些值应同时观察状态停留分布，而不只是最终 IoU。

