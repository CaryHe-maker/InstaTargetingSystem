# 控制器模块

`DepthAwareTrackController` 是状态和逐帧结果的唯一写入者。它把多帧运动预测、固定视域搜索、HiT 推理、候选融合、状态机和模板策略组织成一个有界的 `FrameTransaction`。

## 主要接口

- `buildInitialization(frame, initialBox)`：生成固定 `120° × 120°` 的初始化视图，并把初始框投影到该视图。
- `commitInitialization(plan, depthSummary)`：建立运动历史并提交第 0 帧。
- `beginFrame(frame)` / `plan(frame)`：根据帧开始时的 `TrackMode` 和多帧预测中心 `c1` 创建本帧搜索计划。
- `consume(plan, observations)`：提交当前轮次；若状态专属路线仍有下一轮，返回 `MoreViewsRequired`。
- `update(plan, observations)`：兼容调用路径，直接提交当前轮次，不隐式创建额外搜索。

## 固定视域与轮次

所有局部和 cubemap `ViewSpec` 都使用最大 `120° × 120°`，输出像素尺寸仍由 `geometry.viewWidthPx/viewHeightPx` 决定，不按目标尺度或轮次缩放。

| 帧开始状态 | ROUND_1 | ROUND_2 | ROUND_3 | 本帧最大视图数 |
|---|---:|---:|---:|---:|
| `TRACKING` | 4 个四角视域，以 `c1` 为中心 | 4 个四角视域，以 ROUND_1 最高置信度框中心为中心，直接结束 | 无 | 8 |
| `UNCERTAIN` | 4 个四角视域 | 4 个四角视域，以 ROUND_1 最高置信度框中心为中心；超过 `SuccessRate` 即结束 | 6 面 cubemap，直接选择最高置信度框结束 | 14 |
| `RECOVERING` | 同 `UNCERTAIN` | 同 `UNCERTAIN` | 同 `UNCERTAIN` | 14 |
| `LOST` | 6 面 cubemap | 4 个四角视域，以 ROUND_1 最高置信度框中心为中心，直接结束 | 无 | 10 |

四角视域的中心相对 seed 在局部切平面偏移 `±40°`；相邻视域覆盖重合为 `1/3`，四视域共同中心区域为 `1/9`。cubemap 固定为 front/right/back/left/up/down 六个方向。

## 事务一致性

控制器校验 `sequenceId`、`frameIndex`、`transactionId`、`attemptIndex`、`stateRevision`、模板修订号和 `viewId` 顺序。重复、乱序、跨帧或超预算响应触发 `ProtocolError`。一帧只在最终轮次调用一次状态转换并提交一个结果；中间轮次不会修改跨帧状态、运动历史或模板。

## 候选评估与输出

`StateEvaluator` 使用 `ProjectedObservation.fusedScore` 作为局部框置信度。第一轮（LOST 除外）允许 `OverlapRate > firstRoundFusionOverlap` 的双框融合；LOST 第一轮及所有后续轮次使用 `OverlapRate > OverlapThreshold`。`OverlapRate` 是球面/ERP 预测框交集面积除以较小框面积，默认 `OverlapThreshold=0.70`。

每个 FuseBox 最多包含两个不同视域的局部框，并以 seam-aware 最小联合框表示。融合置信度为：

```text
1 - ((2 - b - a) * (1 - y) / 2)
```

`FusionSourceMinConfidence=0.80` 是源框门控：只有两个源框置信度都大于等于该值时，FuseBox 才能被归类为 `RELIABLE_FUSED`。该门控不阻止 FuseBox 生成，也不阻止最终轮次输出弱候选。

最终证据分为 `RELIABLE_FUSED`、`RELIABLE_SINGLE`、`WEAK`、`MISSING`。第一轮只有可靠 FuseBox（融合重合超过 `OverlapThreshold`、置信度超过 `SuccessRate` 且源框门控通过）可提前结束；第二轮的 `UNCERTAIN/RECOVERING` 在最高候选置信度超过 `SuccessRate` 时结束，`TRACKING/LOST` 第二轮无条件结束；第三轮无条件选择最高候选。

可靠观测使用 `OBSERVED_CONFIRMED` 或找回时的 `OBSERVED_REACQUIRED`。最终轮次即使只有弱候选也输出 `OBSERVED_WEAK_BLEND`；没有候选才输出 `MOTION_PREDICTED`。弱候选和运动预测不会更新可靠运动历史或模板。

## 状态转换

- `TRACKING`：可靠证据保持 `TRACKING`；弱或缺失进入 `UNCERTAIN`。
- `UNCERTAIN`：可靠证据回到 `TRACKING`；弱证据按 patience 计数，缺失立即进入 `LOST`，弱证据达到 patience 后进入 `LOST`。
- `LOST`：可靠 FuseBox 直接找回到 `TRACKING` 并重置运动历史；可靠单框进入 `RECOVERING`；其余保持 `LOST`。
- `RECOVERING`：可靠 FuseBox 回到 `TRACKING`；可靠单框需连续达到 `recoverConfirmFrames` 才回到 `TRACKING`，否则保持 `RECOVERING`；弱或缺失回到 `LOST`。

模板只在可靠跟踪/找回提交后按 `TemplatePolicy` 更新；弱候选、运动 fallback 和未确认找回均保持模板不变。

## 深度行为

深度摘要仅在 `depth.enabled` 且帧提供有效深度时计算。RGB-only 配置的 `depthProcessor` 和 `depthEncoder` 为空，但控制器保持相同的可选深度字段和状态语义。
