# 状态机算法

## 状态

内部状态只有 `INIT`、`TRACKING`、`UNCERTAIN`、`LOST`。初始化提交后公开状态为 `TRACKING`；公开 `TrackStatus` 不单独输出 `INIT`。

`RECOVERING` 已移除。LOST 找回时仍保留“是否接受测量”的独立判断和运动历史重置，但不再增加公开中间状态。

## StateScore 与 ScoreGroup

每个状态机维护一个寿命相同、容量为 10 的 `ScoreGroup`，保存最终候选的 StateScore。StateScore 是当前最终候选的 SingleScore；若最终候选是 Fusor 产生的融合框，则使用当前融合公式产生的融合分数。

状态转移只读取 StateScore，测量提交、运动历史更新和模板更新仍由 Controller 的独立接受门控决定。弱候选可以产生预测/弱输出，但不会写入公开测量历史。

第一、第二个 tracking 状态无条件使下一状态为 `TRACKING`，并在状态结束后记录分数。第三个状态开始按以下顺序选择下一状态：

- 第二个分数大于第一个分数：下一状态为 `TRACKING`，否则为 `UNCERTAIN`。
- 第 3 至第 10 个状态：`UT = 0.5 * max + 0.5 * min`，`LT = 0.2 * max + 0.8 * min`。
- 第 11 个状态起：按 ScoreGroup 降序排列，`UT` 取第 5 大数据，`LT` 取第 8 大数据。

每次帧事务提交后才把当前 StateScore 加入 ScoreGroup。无候选使用 0；全零预热窗口不会停留在 `TRACKING`，而会进入 `LOST`，避免 `score >= UT == 0` 造成死循环。

统一转移规则为：

```text
StateScore >= UT       -> TRACKING
LT <= StateScore < UT  -> UNCERTAIN
StateScore < LT        -> LOST
```

## 证据与提交

StateEvaluator 仍输出可靠融合、可靠单框、弱和缺失等证据，供 Controller 判断 `acceptMeasurement`。这项判断不能被 StateScore-only 状态选择替代：只有被接受的测量才更新当前 bbox/BFoV、运动样本或模板。

LOST 的可靠融合或达到候选质量门控的单框可以直接回到 `TRACKING` 并重置运动历史；低质量候选仍可让下一状态由 StateScore 决定，但结果保持 `valid=False`。

## 模板与事务

同一帧的第一、第二轮使用相同的模板特征快照内容；每轮命令仍有独立且严格递增的 expectedRevision，第二轮强制 `KEEP`。FrameTransaction 在最终状态选择前暂存轮次结果，只有一次提交，防止中间轮污染状态。
