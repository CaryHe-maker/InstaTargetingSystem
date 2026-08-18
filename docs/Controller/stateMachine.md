# 状态机算法

## 状态

内部类型保留 `INIT`、`TRACKING`、`UNCERTAIN`、`LOST`。初始化提交后公开状态为 `TRACKING`；公开 `TrackStatus` 不单独输出 `INIT`。当前实验的正常评分线程只会选择 `TRACKING` 或 `UNCERTAIN`，不会自动进入 `LOST`。

`RECOVERING` 已移除。LOST 的状态类型、10 视图 planner、找回接受判断和运动历史重置组件仍保留，供显式调用、兼容和后续实验使用；它们不由当前 StateScore 转移自动触发。

## StateScore 与 ScoreGroup

每个状态机维护一个寿命相同、容量为 10 的 `ScoreGroup`，保存最终候选的 StateScore。StateScore 是当前最终候选的 SingleScore；若最终候选是 Fusor 产生的融合框，则使用当前融合公式产生的融合分数。

状态转移只读取 StateScore，测量提交和运动历史更新仍由 Controller 的独立接受门控决定。模板固定为第 0 帧 anchor，不参与稳定状态更新。弱候选可以产生预测/弱输出，但不会写入公开测量历史。

第一、第二个 tracking 状态无条件使下一状态为 `TRACKING`，并在状态结束后记录分数。第三个状态开始按以下顺序选择下一状态：

- 第二个分数大于第一个分数：下一状态为 `TRACKING`，否则为 `UNCERTAIN`。
- 第 3 至第 10 个状态：`UT = 0.5 * max + 0.5 * min`，`LT = 0.2 * max + 0.8 * min`。
- 第 11 个状态起：按 ScoreGroup 降序排列，`UT` 取第 5 大数据，`LT` 取第 8 大数据。

每次帧事务提交后才把当前 StateScore 加入 ScoreGroup。无候选使用 0；全零预热窗口不会停留在 `TRACKING`，而会进入 `UNCERTAIN` 并记录 `HARD_MISS`，避免 `score >= UT == 0` 造成死循环。

统一转移规则为：

```text
StateScore >= UT       -> TRACKING
LT <= StateScore < UT  -> UNCERTAIN
StateScore < LT        -> UNCERTAIN (reason=HARD_MISS)
```

因此当前正常线程不会发布新的 `LOST` 结果；低于 LT 与全零缺失都继续执行 UNCERTAIN 的两轮扩大局部搜索。`LT` 和 HARD_MISS 诊断仍保留，便于比较实验结果。

## 证据与提交

StateEvaluator 仍输出可靠融合、可靠单框、弱和缺失等证据，供 Controller 判断 `acceptMeasurement`。这项判断不能被 StateScore-only 状态选择替代：只有被接受的测量才更新当前 bbox/BFoV、运动样本或模板。

若兼容调用显式构造 LOST 状态，可靠融合或达到候选质量门控的单框仍可直接回到 `TRACKING` 并重置运动历史；这条保留路径不代表正常评分线程能够进入 LOST。

## 模板与事务

同一帧的第一、第二轮使用相同的模板特征快照内容；每轮命令仍有独立且严格递增的 expectedRevision，第二轮强制 `KEEP`。FrameTransaction 在最终状态选择前暂存轮次结果，只有一次提交，防止中间轮污染状态。
