# 逐帧处理流程

运行时采用单线程、同步、严格有序的帧事务。每帧只进行一次跨帧状态提交和一次结果发布。

1. 第 0 帧根据初始框生成 `InitializationPlan`，使用固定 `120° × 120°` 视域编码模板。
2. 后续帧由多帧运动预测模块生成中心 `c1`，控制器读取帧开始时的 `TrackMode` 并固定本帧路线。
3. `RecoveryPlanner` 生成本轮 `ViewSpec`：四角路线使用相对 seed 的 `±40°` 中心偏移；全局路线使用六面 cubemap；每个视域均为 `120° × 120°`。
4. 几何模块裁剪局部视图，HiT 后端逐视域推理；RGB-D 路线同步处理深度伪彩色视图。
5. 运行时将局部框回投影为 ERP/BFoV，并附加 `fusedScore`、运动、尺度、深度和视域标识。
6. `StateEvaluator` 按本轮阈值执行确定性的一对一双框融合，保留原始局部框和 FuseBox，选择全局最高候选并分类证据。
7. 若当前轮次未满足该状态的提前结束条件，控制器返回 `MoreViewsRequired`，驱动请求下一轮；路线最多为 `TRACKING=2`、`UNCERTAIN=3`、`RECOVERING=3`、`LOST=2` 轮。
8. 最终轮次直接提交候选最高框；无候选时提交运动预测框。此时控制器只调用一次状态机转换，并更新运动历史、恢复计数和模板策略。
9. 结果写入 sink，随后读取下一帧。

## 状态与输出

公共状态为 `TRACKING`、`UNCERTAIN`、`RECOVERING` 和 `LOST`。最终证据为：

- `RELIABLE_FUSED`：FuseBox 的融合重合率超过 `OverlapThreshold`、置信度超过 `SuccessRate`，且两个源框均达到 `FusionSourceMinConfidence`。
- `RELIABLE_SINGLE`：单个局部框置信度超过 `SuccessRate`。
- `WEAK`：有候选但未通过可靠门控；最终轮次仍可输出，来源为 `OBSERVED_WEAK_BLEND`。
- `MISSING`：没有候选，来源为 `MOTION_PREDICTED` 且 `valid=False`。

弱候选、未确认的 LOST 单框和运动 fallback 不更新可靠运动历史或模板。可靠 FuseBox 找回会重置运动历史；RECOVERING 的可靠单框必须连续确认后才回到 TRACKING。

可视化只读取已经提交的中间数据并写 PNG，不向控制器返回信息，因此不会改变事务决策。
