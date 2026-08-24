# InstaTargetingSystem 增强实验总报告

## 1. 范围与总判断

本文汇总 `docs/` 中三份实测报告：

- [ERPChangeReport.md](ERPChangeReport.md)：第二轮直接从 ERP 原图裁图。
- [FuseScoreReport.md](FuseScoreReport.md)：bbox refinement、尺度约束和时序形状先验。
- [NewPicReport.md](NewPicReport.md)：受控更换 HiT recent template。

三轮实验均使用 validation 数据，没有读取 holdout。不同报告使用的代码阶段、生产基线和序列集合并不完全相同，因此本文只比较各报告内部相对其对应基线的增量，不把不同报告的绝对指标直接排序。

总判断是：**目前没有任何实验变体具备直接替换 production V2 默认路径的证据。** 已实现的实验机制可以保留，部分方案值得继续研究，但生产默认仍应保持共享 V2 路径和 frame-0 anchor。

| 实验方向 | 最好信号 | 主要风险 | 当前决定 |
| --- | --- | --- | --- |
| ERP direct crop | `2x_relaxed` 提高 Success@0.5；4x 在单条 `seq_0045` 改善 IoU/loss | `seq_0017` 严重回归，visible loss 大增，触发率接近全替换 | 全部不晋级 |
| Delta refinement | `seq_0045` ERP IoU `+0.0118` | 候选级 local IoU 没有稳定提升，真实序列回归 | 仅作为 v2 研究起点 |
| Quality-aware refinement | `seq_0017/0045` IoU 提升 | `seq_0036` 与 success 指标回归，pilot 训练不足 | 不启用 |
| Tracking scale clamp | 三序列宏平均 IoU 有正信号，尺寸误差下降 | `seq_0045` absent FPR 恶化，`seq_0036` 回归 | 仅保留为带 presence gate 的候选 |
| Temporal shape prior | `seq_0045` 有局部收益 | 错误框会形成自反馈，`seq_0017` 明显回归 | 停止当前实现 |
| Strict template | `seq_0045` IoU/loss 明显改善 | 加权 IoU 下降，`seq_0036` 严重回归 | 不默认启用 |
| Relaxed template | 宏平均最高，加权 IoU微增 | 收益仅约 `+0.0010`，多序列下降，回滚和成本高 | 不默认启用 |

## 2. ERP 第二轮直接裁图

### 2.1 设计

实验尝试在第二轮绕过常规 Geometry Type1 透视视图，直接围绕第一轮结果从 ERP 图像裁取 `2x` 或 `4x` 区域，再缩放至 HiT 输入尺寸。5-seq 预筛包含 `sim/0045`、`sim/0017`、`real/0005`、`sim/0078`、`sim/0010`。

### 2.2 聚合结果

| 变体 | Macro Mean IoU | 相对基线 | Macro loss | Micro Mean IoU | Micro loss | Absent FPR | Pooled P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Production | 0.326417 | - | 15.25% | 0.294693 | 19.11% | 56.07% | 314.23 ms |
| `2x_strict` | 0.277825 | -0.048592 | 31.45% | 0.269940 | 31.80% | 0.00% | 353.24 ms |
| `2x_relaxed` | 0.327600 | +0.001182 | 28.78% | 0.299470 | 32.58% | 4.62% | 334.04 ms |

`2x_relaxed` 的平均 IoU 和 Success@0.5 有正信号，但不能抵消 loss 的明显恶化。核心硬门槛 `sim/0017` 上，strict/relaxed 的 Mean IoU 分别下降 `0.233708` 和 `0.168508`，loss 分别增加 `53.24` 和 `37.58` 个百分点。

两条 2x 路径的整体触发率为 `69.11%` 和 `74.67%`；在部分序列接近 100%，实质上已经接近替换整个第二轮。仅判断 crop 是否越界不能代表跟踪稳定性。

### 2.3 4x 诊断

4x 只在 `sim/0045` 运行：Mean IoU `0.176281 → 0.206975`，loss `42.16% → 35.12%`，absent FPR `56.07% → 33.53%`。但该实验同时修改了 crop 尺度、触发条件和第一轮 FOV，且没有通过 `sim/0017` 硬门槛，不能作为单变量证据。

### 2.4 决定

- 保留常规 Geometry Type1 第二轮路径。
- 回退 `2x_strict` 和 `2x_relaxed`。
- 4x 只保留诊断记录，不晋级。
- 在稳定性 gate 能保护 `sim/0017` 前，不继续扩大 crop 倍数或运行完整 validation/holdout。

## 3. FuseScore 与 bbox 增强

### 3.1 设计和数据限制

该报告在 production2 基线上比较三条 validation sequence：`sim/0017`、`sim/0045`、`real/0036`。

- `iou_refine_head`：用 HiT embedding、初始框和 corner stability 预测 bbox delta。
- `iou_refine_quality_aware`：额外使用 corner heatmap 位置、方差和峰值特征。
- `tracking_scale_clamp`：TRACKING 状态下限制 width/height 到上一框的 `0.7~1.3`。
- `fuse_temporal_shape_prior`：按历史框形状相似度增加候选置信度。

两个 refinement checkpoint 都只是 100-step pilot，不能视为完整训练结果。校准使用独立 calibration split，未读取 holdout。

### 3.2 结果摘要

下表宏平均仅用于压缩三条序列的方向性，不替代报告中的逐序列 gate：

| 方案 | Macro ERP IoU | 相对基线 | Macro loss | `seq_0045` absent FPR | 判断 |
| --- | ---: | ---: | ---: | ---: | --- |
| V2 baseline | 0.248091 | - | 36.73% | 50.29% | 生产对照 |
| Delta refinement | 0.245644 | -0.002447 | 38.27% | 45.09% | 框质量改善不稳定 |
| Quality-aware | 0.254130 | +0.006039 | 33.28% | 36.99% | `real/0036` 和 success 回归 |
| Scale clamp | 0.257995 | +0.009904 | 33.27% | 59.54% | 定位有收益，但 absent gate 失败 |
| Temporal prior | 0.236726 | -0.011365 | 37.37% | 43.93% | 自反馈风险明确 |

虽然 quality-aware 和 scale clamp 的宏平均较高，但预设门槛要求 success@0.5 不下降超过 0.5 pp、loss/FPR 不恶化超过 1 pp，并保护 `sim/0017` 与 `sim/0045` absent 场景；四组均至少违反一项。

### 3.3 关键解释

- Delta head 的候选级 local IoU 没有稳定改善，端到端变化主要来自后续状态轨迹，不能解释为 refinement 已学会更准确的框。
- Quality-aware 在 `sim/0017` 提高 IoU，但 success@0.5 下降；在 `real/0036` 的 IoU、success 和 loss 都回归。
- Scale clamp 对尺寸 P95 有效，但可能让 absent 帧上的错误框持续得更稳定，因此必须与 presence/absent gate 联合。
- Temporal prior 使用已提交框作为 reference；错误提交会提高后续相似错误框的分数，形成自增强闭环。

### 3.4 决定

生产默认保持 V2。若继续推进，优先级为 geometry-aware delta acceptance，其次是带 presence gate 的 scale clamp；quality-aware 需完整训练后重评，当前 temporal prior 不继续使用。

## 4. 受控更换 HiT 模板

### 4.1 已验证语义

该实验在 production2 中选择 5 组：`real/0005`、`real/0018`、`real/0036`、`sim/0010`、`sim/0045`。实现并实际执行了以下约束：

- frame-0 anchor 图像和特征永久保留。
- 候选只来自已提交、`valid=true`、measurement accepted 的结果。
- 下一帧使用 anchor 公开分支和候选 shadow 分支验证。
- 验证帧公开 anchor 结果；候选通过后从再下一帧使用。
- 所有候选始终与 anchor 验证，不形成动态模板链。
- shadow Controller 为正式状态的深拷贝，不写正式 Controller、运动历史或结果。
- 连续 3 帧 UNCERTAIN 回滚 anchor。

### 4.2 精度结果

| Sequence | Baseline IoU | Strict IoU | Strict Δ | Relaxed IoU | Relaxed Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| `real/0005` | 0.3396 | 0.3398 | +0.0003 | 0.3518 | +0.0122 |
| `real/0018` | 0.2380 | 0.2350 | -0.0029 | 0.2225 | -0.0155 |
| `real/0036` | 0.3175 | 0.2752 | -0.0423 | 0.3129 | -0.0046 |
| `sim/0010` | 0.4704 | 0.4743 | +0.0039 | 0.4629 | -0.0075 |
| `sim/0045` | 0.1927 | 0.2539 | +0.0613 | 0.2539 | +0.0612 |

| 聚合 | Baseline | Strict | Relaxed |
| --- | ---: | ---: | ---: |
| Macro ERP IoU | 0.3116 | 0.3157 | 0.3208 |
| 8,134 visible 帧加权 ERP IoU | 0.2740 | 0.2712 | 0.2750 |

relaxed 的加权收益只有约 `+0.0010`，并在三条序列下降。strict 在 `sim/0045` 收益很大，但 `real/0036` 的下降几乎抵消了这一收益，加权结果反而低于 baseline。

### 4.3 更新行为与成本

| 变体 | 候选生成 | 提升 | 拒绝 | 回滚 | 非 anchor 帧 | Shadow 额外耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Strict | 864 | 503 | 361 | 196 | 1,526 | 68.09 s |
| Relaxed | 1,356 | 980 | 376 | 291 | 2,186 | 106.87 s |

所有验证延迟均为 1 帧。事件记录包含 PhotoChangeRate、验证 IoU、anchor/候选置信度和框差异、动态模板持续时间以及 shadow 分阶段成本。功能、隔离和时序已经通过真实 CUDA/HiT 推理和定向测试验证，不是仅记录状态的占位实现。

### 4.4 决定

模板实验基础设施可以保留，但 strict 和 relaxed 都不应成为默认策略。当前 StateScore/top-2 条件不能可靠预测动态模板对未来帧的真实收益，频繁回滚和 shadow 成本也说明 relaxed gate 过宽。

## 5. 跨实验共同规律

### 5.1 平均 IoU 不能单独决定晋级

多个方案出现“平均 IoU 提升但 loss、success 或 absent FPR 恶化”：ERP `2x_relaxed`、scale clamp 和 relaxed template 都属于这一类。生产晋级必须同时检查：

- ERP Mean IoU、Spherical IoU 和 Success@0.5；
- visible tracking loss；
- absent FPR；
- P50/P95/P99 和额外 forward/crop 成本；
- 至少一条真实序列和一条已知硬回归序列。

### 5.2 `sim/0045` 是收益场景，不是充分证据

ERP crop、多个 FuseScore 变体和动态模板都能在 `sim/0045` 产生收益，但 absent FPR、visible loss 和其他序列的变化方向不同。该序列适合检验恢复、消失目标和外观变化，不适合单独决定生产晋级。

### 5.3 `sim/0017` 和 `real/0036` 应作为早停门槛

`sim/0017` 能暴露 ERP direct crop 和 temporal prior 的严重漂移；`real/0036` 能暴露 refinement、scale clamp 和 strict template 的真实场景回归。后续实验应先通过这两条序列，再扩大到完整 validation。

### 5.4 自反馈是当前主要风险

直接 ERP 路径高触发率、shape prior 使用历史错误框、动态模板提升后影响未来观测，这三类方案都会改变后续状态分布。一帧上的局部 gate 通过不等于长期安全，必须增加持续收益检查、独立回滚依据和首次漂移分析。

### 5.5 分支 gate 需要因果可观测信号

仅使用 crop 越界、StateScore 排名或形状相似度不足以判断未来收益。更有价值的 gate 应组合：

- 连续 measurement acceptance 和状态稳定性；
- presence/absent probability；
- 尺度创新量、aspect ratio 变化和目标占 crop 比例；
- anchor/候选在验证帧的置信度、中心和尺度差异；
- 最近若干帧的公开结果收益，而不是单帧自评。

## 6. 统一生产决策

当前建议如下：

1. 生产默认保持 production V2、常规 Geometry Type1 第二轮和 frame-0 anchor。
2. 不启用 ERP direct crop、quality-aware pilot、temporal shape prior、strict template 或 relaxed template。
3. 保留受控模板的 shadow 验证和事件记录能力，作为后续实验基础设施。
4. 优先研究 geometry-aware delta acceptance；候选必须有 predicted-IoU/corner gate，并证明候选级 local/ERP IoU 稳定改善。
5. Scale clamp 仅在 presence 可靠、连续稳定且目标可见时研究，禁止在 absent/低 presence 状态提交。
6. 新方案先跑 `sim/0017` 和 `real/0036` 硬门槛，再跑 `sim/0045` 检查 absent 行为，最后才扩大 validation。
7. 只有加权 ERP IoU、tracking loss、absent FPR 和 P95 延迟同时不劣于基线，且无单序列硬回归，才进入生产候选。

## 7. 证据边界

- ERP 报告是 5-seq 预筛，4x 仅单序列，不代表完整 validation。
- FuseScore 报告只有 3 条序列，refinement 仅 100-step pilot。
- 模板报告覆盖 5 条完整序列，但 relaxed 的加权收益极小且回滚频繁。
- 三份报告均未读取 holdout；在模型、gate 和默认策略冻结前不应使用 holdout 反复选择方案。
- 延迟结果来自不同运行批次和实现阶段，只能在同一报告、同一运行条件下解释。

综上，当前增强工作的主要成果是识别了可靠的失败模式和建立了可审计实验基础设施，而不是得到可直接上线的算法开关。下一阶段应收紧变量隔离和晋级门槛，优先消除漂移、自反馈与 absent 误提交。
