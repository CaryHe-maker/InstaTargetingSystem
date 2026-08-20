# 推测流水线与 HiT 后端适配蓝本

> 状态更新：Stage 3、独立校准和阶段 4 A4 已完成；阶段 5 推测/效率集成尚未开始。本文其余内容保留为阶段 5 设计约束。
>
> 本文中的阈值和回退比例都是 **provisional（临时预测值）**，除非经过独立 calibration split 和端到端验证，不得宣称为生产参数。

## 1. 文档目的

本项目当前每个普通跟踪帧使用两轮局部视图搜索：

```text
Round 1: 4 views -> HiT batch 4 -> Controller 选择搜索中心
Round 2: 4 views -> HiT batch 4 -> 两轮观测统一 Fusor -> 提交结果
```

拟实施的优化是跨帧推测流水线：

```text
R1(t) -> 规划 R2(t) 和推测的 R1(t+1)

稳态后端批次：
    [R2(t), R1(t+1)] -> 一次 HiT batch 8
    [R2(t+1), R1(t+2)] -> 一次 HiT batch 8
```

设计目标：

1. 每个输入帧仍保留 4+4 视图预算和第二轮精细框判断。
2. Round 1 主要负责确定下一帧搜索方向，不直接替代 Round 2 的形状判断。
3. Round 2 仍使用现有 Fusor 和两轮观测统一排序/融合语义。
4. 在不牺牲最终精度的前提下，把两次 batch 4 变成稳态的一次 batch 8。
5. 推测失败时可以丢弃推测结果并回退到现有串行 4+4 路径。

## 2. 当前基线与事实

实施者必须以当前代码和实验产物为准，不得凭 HiT 官方单视图 FPS 推断本系统速度。

当前系统事实：

- 运行主线是 `ERP RGB -> Geometry 局部透视视图 -> HiT -> 球面/ERP 回投 -> Controller`。
- 正常线程只在 `TRACKING` 和 `UNCERTAIN` 之间运行；当前不会由普通状态机自动进入 `LOST`。
- `TRACKING` 和 `UNCERTAIN` 都使用两轮 `4 + 4`；保留的显式 `LOST` 组件使用一次 `6 + 4`。
- 第二轮中心依赖第一轮 Fusor 结果。
- HiT 模板是第 0 帧 anchor；`TemplateCache` 当前保存预处理模板张量，不等于完整 backbone 特征缓存。
- `ProjectedObservation` 才是跨局部视图比较的统一球面/ERP语义。
- 当前 Controller 使用 `singleScore` 排序、融合和来源门限。
- 当前 `singleScore = 0.50 * appearanceProbability + 0.50 * effectiveMotionProbability`，权重来自 Stage 3 校准产物。
- 当前 `appearanceProbability` 对 Stage 3 `presence*predictedIoU` 使用 checkpoint 绑定的 Beta Calibration。
- 当前固定配置要求 `tracking.maxAttemptsPerFrame = 2`，总视图预算至少为 12；不能把生产配置直接改成 8 来掩盖预算问题。

已知的非 holdout A/B 事实：

- legacy 权重在 validation 序列 `train_sim/seq_0010` 上完成 450 帧运行。
- legacy circular ERP mean IoU 约为 `0.330`，P95 约为 `386.8 ms`。
- Stage 2 的历史结果仅用于解释为何 Stage 3 必须重新校准，不参与当前生产加载或阈值选择。

Stage 3 训练期间的工程约束：

- Stage 3 使用 `E:\InstaTargetingSystemTraining\checkpoints\stage3` 输出。
- 训练数据为 `E:\NewDownload\train\manifest.jsonl`。
- Stage 3 运行期间不得修改训练代码、训练配置、manifest、模型结构或依赖环境。
- 不要在训练中的工作区直接做可能影响 resume 的修改；适配工作应使用独立 worktree/分支。

## 3. 允许提前实施的范围

以下工作可以在 Stage 3 训练期间提前完成，但必须默认关闭并保持旧路径可运行。

### 3.1 只实现机制，不冻结生产参数

可以实现：

- speculative state（推测状态）数据结构。
- 跨帧任务标识和结果路由。
- `R2(t)` 与 `R1(t+1)` 的 batch 合并调度。
- 推测结果的接受、失效和重跑状态机。
- 推测统计字段和诊断日志。
- 串行 4+4 fallback。
- 单元测试、伪造 session 测试和离线回放工具。

不能在没有 calibration 数据时冻结：

- `candidateMinScore`。
- `fusionSourceMinConfidence`。
- 外观/运动组合权重。
- 推测中心差距阈值。
- 推测尺度差距阈值。
- 目标覆盖率阈值。
- 生产回退率目标以外的任何“优化成功”判定。

### 3.2 允许新增配置，但必须是独立且默认关闭的

如果需要配置字段，应放在推理配置的明确子节，例如：

```yaml
speculativePipeline:
  enabled: false
  batchMergeEnabled: false
  maxRollbackRate: 0.20
  centerGapRatio: 0.50
  logScaleGap: 0.25
  minimumDirectionConfidence: 0.80
```

以上数值只是起始实验值，不是发布值。严格 schema、配置文档和测试必须同步修改。不能把训练字段塞进 `AppConfig` 的既有模块，不能绕过 schema 直接读取任意 YAML 键。

## 4. 严格禁止的修改

### 4.1 训练隔离

Stage 3 运行期间禁止修改：

- `src/instatarget/training/**`。
- `src/instatarget/core/config.py` 中 TrainingConfig 解析逻辑。
- `configs/train_stage3.yaml`。
- `E:\NewDownload\train\manifest.jsonl` 或其内容。
- `src/instatarget/training/model.py`、loss、dataset、optimizer、scheduler。
- Python、PyTorch、CUDA、torchvision、timm 或其他训练依赖。
- Stage 2/Stage 3 checkpoint 文件。

如果必须修复训练 bug，应停止当前实验，复制实验目录并以新实验编号重新开始；不得在原 Stage 3 目录中混合不同代码版本。

### 4.2 生产算法语义禁止破坏

不得：

- 让 Round 1 直接更新正式运动历史。
- 让 Round 1 直接更新公开 bbox/BFoV、状态分数、模板或 Controller revision。
- 用 Round 1 的粗框替代 Round 2 的精细形状判断。
- 因为流水线而删除第二轮、显式 fallback 或完整恢复路径。
- 用 Controller 最终是否接受作为 presence/quality 训练标签。
- 把运动预测写入 ground truth 或训练数据。
- 修改 Geometry 的 seam-aware ERP 交集、球面投影和 BFoV 语义来换取速度。
- 在未校准前手调映射或通过提高阈值掩盖分数分布问题。
- 在 Stage 3/calibration 完成前读取、调参或评估 final holdout。
- 把平均 FPS 作为唯一性能结论；必须同时报告 P50/P95/P99。

## 5. 推测流水线的状态模型

流水线必须区分两类状态：

### 5.1 正式状态（committed state）

正式状态由已完成 Round 2 的最终候选提交，包含：

- 当前正式 `TrackResult`。
- 当前公开状态和 `StateScore` 历史。
- 正式运动历史和 MotionPredictor 窗口。
- 正式模板 revision 和模板内容。
- 正式 FrameTransaction 提交序号。

只有正式状态可以影响下一帧的公开输出和生产运动历史。

### 5.2 推测状态（speculative state）

推测状态只用于提前规划下一帧 Round 1，包含：

- 推测 frame index。
- 推测方向中心。
- 规划时使用的正式尺度快照。
- 推测候选置信度和方向质量。
- 生成该推测的正式状态 revision。
- generation/token，用于识别过期任务。

推测状态不能：

- 写入正式运动历史。
- 写入 StateMachine 的 score group。
- 修改当前公开 bbox/BFoV。
- 修改模板或 template revision。
- 直接作为最终结果写入 sink。

## 6. 推荐执行顺序

### 6.1 初始化帧

初始化仍完全沿用现有流程：

1. 读取第 0 帧。
2. 使用初始 bbox 建立第 0 帧 anchor。
3. 提交初始化结果。
4. 不创建跨帧推测任务，直到正式状态可用。

### 6.2 帧 t 的第一轮

1. 基于正式状态生成 `R1(t)` 的 4 个 ViewSpec。
2. 执行 HiT batch 4。
3. 对观测执行现有的 appearance calibration、Geometry 回投和诊断记录。
4. 只用第一轮结果估计“方向候选”，不要把第一轮形状当成最终框。
5. 使用方向候选规划 `R2(t)`。
6. 同时基于该方向候选和正式尺度快照，生成推测的 `R1(t+1)`。

### 6.3 稳态后端批次

当 `R2(t)` 和推测的 `R1(t+1)` 都已准备好时，后端可以提交一个 batch 8：

```text
batch slot 0..3: R2(t)
batch slot 4..7: speculative R1(t+1)
```

必须在每个 batch slot 附带不可变的：

```text
sequenceId, frameIndex, attemptIndex, viewId, generation, role
```

输出不得按“完成顺序”解释，只能按上述身份路由。`role` 至少区分 `round2_shape` 和 `speculative_round1_direction`。

### 6.4 Round 2 完成后的正式提交

R2(t) 返回后：

1. 将 R1(t) 与 R2(t) 的全部 ProjectedObservation 放入同一候选池。
2. 仍由现有 Fusor 按 `singleScore` 排序和融合。
3. 仍执行现有 `candidateMinScore` 和来源门限语义；阈值未校准前不改。
4. 得到最终 `TrackResult` 后，原子更新正式状态。
5. 用正式结果与已经运行的推测 R1(t+1) 比较。
6. 如果推测仍有效，保留其局部视图/观测以继续规划；如果失效，标记过期并重跑 R1(t+1)。

## 7. 推测有效性与回退规则

### 7.1 必须检查的条件

推测 R1(t+1) 只有在全部必要条件满足时才可接受：

- R1(t) 至少有有效方向候选。
- R1(t) 的候选置信度达到临时方向门槛。
- R1(t) 与 R2(t) 的最终中心差距未超过归一化阈值。
- R1(t) 与最终结果的尺度差距未超过阈值，或尺度仍沿用正式尺度且未越界。
- R1(t+1) 的四个视图覆盖正式修正后的预测中心。
- 推测 generation 与正式状态 revision 匹配。
- 推测任务没有被更晚的正式提交标记为 stale。
- 所有 bbox、BFoV、概率和投影字段均有限且符合协议。

### 7.2 建议的归一化中心差距

不要只使用固定角度阈值。建议记录：

```text
centerGapRatio =
    greatCircleDistance(speculativeCenter, committedCenter)
    / max(predictedTargetAngularSize, motionUncertainty, epsilon)
```

临时起始值：`centerGapRatio = 0.50`。该值只用于早期回放实验，必须在 calibration split 重新选择。

### 7.3 建议的尺度差距

使用 log 尺度差：

```text
logScaleGap = max(
    abs(log(speculativeWidth / committedWidth)),
    abs(log(speculativeHeight / committedHeight))
)
```

临时起始值：`0.25`。不得用线性像素差替代球面角尺寸或 log 尺度语义。

### 7.4 必须回退的情况

遇到以下任一情况，必须丢弃推测结果并回退：

- R1 没有有效候选。
- R1/R2 中心差距超过阈值。
- 尺度差距超过阈值且下一轮覆盖不足。
- 目标可能脱离下一帧 R1 四视图覆盖区域。
- R1 置信度、bbox、BFoV 或预测输出非有限。
- generation/revision 不匹配。
- 后端输出顺序或数量校验失败。
- 正式 Controller 进入显式 LOST 或恢复路径。
- 目标缺失、遮挡或场景变化导致方向可靠性不足。

回退时必须：

1. 标记推测任务为 invalid/stale。
2. 禁止其输出写入正式状态。
3. 用正式状态重新规划并执行 R1(t+1)。
4. 视预算允许时执行正常 R2(t+1)。
5. 记录唯一回退原因，不得静默吞掉。

### 7.5 回退率目标

`rollbackRate <= 0.20` 是实验目标，不是可以通过强行接受推测结果达到的硬编码约束。

如果真实数据表明回退率超过 20%，应优先收紧推测接受条件或退回串行路径，不得为了满足统计比例而接受覆盖不足的推测结果。

## 8. 事务、revision 和并发约束

当前 `FrameTransaction` 是原子提交模型。流水线适配必须扩展它，而不能绕过它。

必须保证：

- 每个 `(sequenceId, frameIndex)` 最多一次正式提交。
- 同一 frame 的 Round 1 和 Round 2 属于同一逻辑事务。
- Round 2 完成前，Round 1 不能更新正式 Controller 状态。
- 模板 revision 由正式提交顺序决定，不能由 GPU batch 完成顺序决定。
- 每个推测任务都有 generation；旧 generation 的迟到输出必须被丢弃。
- sequence 结束时必须取消所有未完成推测任务并释放资源。
- sink 只接收正式 `TrackResult`，不接收局部推测结果。
- source 的 frame 顺序不因后台批处理改变。
- 一个 frame 的推测输出不能跨 sequence 使用。

推荐任务身份：

```text
TaskKey = (
    sequenceId,
    frameIndex,
    attemptIndex,
    viewId,
    generation,
    role,
)
```

## 9. 后端 batch 与显存要求

实现者必须先测量 batch 4、batch 8 和 batch 10，再决定是否合并。不能假设 batch 8 一定比两个 batch 4 快。

至少记录：

- CUDA Event forward time。
- CPU crop time。
- CPU→GPU copy time。
- preprocessing time。
- projection 和 Controller time。
- batch size、view role、frame index。
- 峰值显存和 OOM 次数。

如果 batch 8 OOM：

- 保留逻辑上的推测流水线。
- 允许调度器退回两个 batch 4。
- 不改变视图数和精度语义。
- 不通过降低输入尺寸、删除第二轮或静默丢视图来规避 OOM。

当前 RTX 4060 Laptop GPU 约 8 GiB。FP16、TensorRT、GPU crop 和模板特征缓存必须分开做 A/B，不能在一次实验中同时改变所有变量。

## 10. provisional 参数表

以下值仅用于搭建和回放，不得用于宣称发布：

| 参数 | provisional 值 | 说明 |
|---|---:|---|
| `speculativePipeline.enabled` | `false` | 默认关闭，串行路径是基线 |
| `speculativePipeline.batchMergeEnabled` | `false` | 未完成 batch 8 profile 前关闭 |
| `maxRollbackRate` | `0.20` | 目标，不是强制接受比例 |
| `centerGapRatio` | `0.50` | 归一化中心差距起点 |
| `logScaleGap` | `0.25` | log 尺度差起点 |
| `minimumDirectionConfidence` | `0.80` | 仅为方向候选起点 |
| `maxSpeculativeAgeFrames` | `1` | 推测不可跨多帧滞留 |

禁止把 provisional 参数写入当前冻结的 Controller 常量，除非它们进入独立配置、日志和实验版本。

## 11. 必须新增的诊断字段

每个普通 frame 至少记录：

- `frameIndex`。
- `round1CompletedAt`、`round2CompletedAt`。
- `round1ViewCount`、`round2ViewCount`。
- `batchSize` 和 batch 内各 role 数量。
- `speculativeGeneration`。
- `speculativeAccepted`。
- `speculativeInvalidated`。
- `rollbackReason`。
- `centerGapRad`、`centerGapRatio`。
- `logScaleGap`。
- `coverageAfterCorrection`。
- `formalStateRevision`。
- `templateRevision`。
- `forwardCount`。
- `cropMs`、`inferMs`、`projectionMs`、`controllerMs`、`totalProcessingMs`。

汇总必须包含：

- 推测接受率。
- 回退率及按原因分组的回退率。
- 推测帧目标覆盖率。
- R1/R2 中心差距分布。
- 正式结果 valid rate。
- 每状态帧数。
- 平均视图数和每帧 forward 数。
- P50/P95/P99 端到端处理时间。
- GPU 利用率、峰值显存和温度。

## 12. 必须覆盖的测试

### 12.1 单元测试

- 推测状态不能写入正式运动历史。
- 推测状态不能更新公开 `TrackResult`。
- generation 不匹配时输出被丢弃。
- revision 不匹配时输出被拒绝。
- 中心差距小于阈值时推测可接受。
- 中心差距超过阈值时触发回退。
- 尺度差距计算使用 log 角尺寸。
- 空 R1、非有限概率、非有限 bbox 都触发回退。
- R1/R2 两轮观测仍由同一 Fusor 候选池统一处理。
- 结果顺序按 TaskKey 恢复，而不是按 batch 返回顺序。
- 同一 frame 不能提交两次。
- sequence 关闭后迟到任务不能写入 sink。

### 12.2 后端测试

- batch 8 的输出顺序与输入 TaskKey 一一对应。
- batch 4 fallback 与 batch 8 的单图输出数值在容差内一致。
- learned presence/quality 字段完整传递到 `LocalObservation`。
- Stage 3 `model` checkpoint 可严格加载；旧 checkpoint 生产兼容路径已删除。
- FP16 非有限输出能按现有规则回退 FP32。
- OOM 或后端异常不会提交半个事务。

### 12.3 Controller/Driver 集成测试

- 正常串行路径在 `enabled=false` 时与改动前结果一致。
- `R2(t)` 和 speculative `R1(t+1)` 的批处理顺序正确。
- R2 修正中心后，下一帧推测被接受或重跑的分支均正确。
- 显式 LOST 直接取消推测并走完整恢复路径。
- sink 仍保持 frameIndex 连续和原子发布。
- time artifact 只统计正式 tracking processing 区间，定义不能因并发而改变。

## 13. 评估顺序

不得跳过以下顺序：

### 阶段 A：机制正确性

- `speculativePipeline.enabled=false`。
- 所有既有测试通过。
- 伪造 backend 返回可控的 R1/R2 中心和尺度。
- 验证接受、失效、重跑、LOST 回退和迟到输出。

### 阶段 B：batch 性能

- 固定 Stage 3 权重、固定 validation 序列、固定配置。
- 比较两个 batch 4 与一个 batch 8。
- 只报告 HiT forward 和端到端 P50/P95/P99。
- 如果 batch 8 节省不足 25%，暂停流水线复杂化，先优化 FP16/GPU crop。

### 阶段 C：Stage 3 完成后 calibration

- 只使用 `calibration` split 拟合外观映射和运动/SingleScore 参数。
- calibration split 不得用于 final holdout 报告。
- 记录模型 hash、manifest hash、代码版本、参数和校准指标。
- 重新确定 `candidateMinScore`、`fusionSourceMinConfidence` 和推测方向门槛。

### 阶段 D：validation Controller A/B

固定模型和校准参数，在相同 validation 序列上比较：

1. legacy 串行 4+4。
2. Stage 3 串行 4+4。
3. Stage 3 推测流水线。

至少报告：

- circular ERP IoU、AUC、success@0.5。
- BFoV spherical IoU。
- 球面中心误差。
- 宽高相对误差。
- 目标缺失误报率。
- valid rate。
- 状态分布。
- 推测接受率和回退率。
- P50/P95/P99。

### 阶段 E：final holdout

只有在模型、校准、Controller 阈值和流水线开关全部冻结后，才允许读取 holdout。

holdout 运行不得用于：

- 继续调阈值。
- 选择模型 checkpoint。
- 选择回退阈值。
- 重新拟合 calibration。
- 决定是否删除 fallback。

## 14. 接受门槛

流水线候选至少满足以下条件才可进入进一步优化：

- 推测接受率目标 `>= 80%`。
- 推测回退率目标 `<= 20%`。
- 接受推测帧的下一轮视图覆盖率 `>= 99%`。
- 相对串行 Stage 3 的 circular ERP mean IoU 下降不超过 1–2%。
- success@0.5 下降不超过 1 个百分点。
- 目标缺失误报率不得恶化。
- P95 不得因推测调度而恶化。
- 所有正式事务仍保持一次提交和严格 frame 顺序。

速度目标仍以项目总目标为准：端到端 P95 不超过 `33 ms`。如果推测流水线只提高平均 FPS、但 P95 仍超过目标，不能称为满足服务要求。

## 15. 推荐实施顺序

```text
保存 Stage 3 实验快照/独立 worktree
 -> 新增默认关闭的 speculative state
 -> TaskKey/generation/revision 路由
 -> 串行回退与事务测试
 -> batch 4/8 profile
 -> 伪造数据接受/回退回放
 -> Stage 3 完成
 -> calibration split 校准
 -> validation 串行/流水线 A/B
 -> 只在指标满足时打开 speculativePipeline
 -> TensorRT FP16/GPU crop 等独立性能实验
 -> 最后冻结并评估 final holdout
```

## 16. 交付检查清单

另一个实施者提交前必须回答“是”：

- [ ] 是否没有修改训练代码、manifest 或正在运行的 Stage 3 实验？
- [ ] 是否使用独立 worktree/分支并记录 commit？
- [ ] 是否默认保持串行 4+4？
- [ ] 是否明确分离 committed state 和 speculative state？
- [ ] 是否没有让 R1 更新正式运动历史、模板或公开结果？
- [ ] 是否实现 generation/revision 和迟到输出丢弃？
- [ ] 是否保留完整 4+4、显式 LOST 和异常 fallback？
- [ ] 是否按 TaskKey 恢复 batch 输出顺序？
- [ ] 是否记录接受率、回退率、回退原因和覆盖率？
- [ ] 是否用 CUDA Event 或同步方式测量 HiT forward？
- [ ] 是否完成 batch 4/8 的显存和延迟比较？
- [ ] 是否通过单元、后端和 Driver 集成测试？
- [ ] 是否在 calibration 前没有调整正式 Controller 阈值？
- [ ] 是否在模型和阈值冻结前没有读取 holdout？
- [ ] 是否报告了精度、误报、状态、视图数、forward 数和 P50/P95/P99？

## 17. 相关文档和代码入口

- [trainingPlan.md](trainingPlan.md)：训练阶段、校准和最终验收规范。
- [Overall/motionProjectionUpgrade.md](Overall/motionProjectionUpgrade.md)：当前运动评分、回投和 SingleScore 语义。
- [Overall/runtimeThread.md](Overall/runtimeThread.md)：当前逐帧处理顺序和计时边界。
- [Controller/stateEvaluator.md](Controller/stateEvaluator.md)：Fusor、两轮候选池和正式提交。
- [Controller/templateAndTransaction.md](Controller/templateAndTransaction.md)：模板 revision 和 FrameTransaction 约束。
- [Controller/viewPlanning.md](Controller/viewPlanning.md)：4 视图规划、FOV 和 LOST 组件。
- [Evaluation/verificationWorkflow.md](Evaluation/verificationWorkflow.md)：回归和指标要求。
- `src/instatarget/app/driver.py`：当前顺序运行组合根。
- `src/instatarget/controller/track_controller.py`：正式状态和帧事务所有权。
- `src/instatarget/controller/state_evaluator.py`：每轮候选评估和最终提交。
- `src/instatarget/tracker/backend.py`：模板、HiT batch 和观测生成。
