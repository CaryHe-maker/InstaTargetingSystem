# LOST 3.1 隔离实验报告

## 结论

本轮完成当前生产对照组和 `3` 种 LOST 状态策略 x `3` 种 LOST ViewSpec，共 `10` 组完整 validation 对比。实验只存在于隔离评估工具和 `artifacts/lost_experiments/full/`，没有接入生产状态机、配置或运行路径，也没有读取 holdout。

**本轮不应把九个候选中的任何一个直接实现到生产。**

- 聚合指标最均衡的是 `immediate_q80 + cube6_adaptive_type1`：mean IoU 从 `0.284803` 升到 `0.293914`，tracking loss rate 从 `21.03%` 降到 `16.08%`，absent FPR 从 `56.07%` 降到 `50.29%`。
- 但该组合在 `train_sim/seq_0017` 上发生严重回归：mean IoU 从 `0.269560` 降到 `0.171812`，loss rate 从 `32.15%` 升到 `53.65%`。总体收益主要来自 `seq_0045`，不能掩盖这一分层退化。
- `rollback_q90 + cube6_adaptive_type1` 取得最高 mean IoU `0.294763`，但回溯 `259` 次，重跑 `518` 帧执行，占全部逻辑帧 `14.77%`；P95 达到 `878.33 ms`，且 `seq_0017` 仍退化，因此不接受回溯实现。
- `hysteresis_q90` 会长时间停留在 LOST。三个 ViewSpec 的 LOST 搜索比例为 `34.48%` 至 `37.62%`，LOST 可见帧 mean IoU 只有 `0.1498` 至 `0.1864`。其中 dual cubemap 的低 absent FPR 主要伴随大量 LOST/无效输出，不能解释为成功找回。
- ViewSpec 层面，`cube6_adaptive_type1` 是唯一值得保留到下一轮的方向；固定 FOV Type1 整体较弱，dual cubemap 计算更重且 absent FPR 不稳定。

因此，3.1 当前决策是：**不冻结 LOST 生产方案；保留自适应两轮 ViewSpec 作为下一轮候选，重新设计触发与恢复提交条件。**

## 实验范围

### 数据与对照口径

- Manifest：`E:\NewDownload\train\manifest.jsonl`
- Dataset root：`E:\NewDownload\train`
- Split：仅 `validation`
- 模型：`models/hit_small_stage3.pth`，FP32
- 配置：`configs/RGBonly.yaml`
- holdout：未读取
- 可见评估帧：`3333`
- absent 帧：`173`
- 非初始化逻辑帧：`3506`

| 序列 | 总帧数 | 非初始化可见帧 | absent 帧 |
| --- | ---: | ---: | ---: |
| `train_real/seq_0005` | 835 | 834 | 0 |
| `train_sim/seq_0010` | 450 | 449 | 0 |
| `train_sim/seq_0017` | 480 | 479 | 0 |
| `train_sim/seq_0075` | 450 | 449 | 0 |
| `train_sim/seq_0045` | 1296 | 1122 | 173 |

旧的 `artifacts/post_training/validation_expanded_summary.json` 只有 `2918` 个可见帧，因为当时 `seq_0005` 和 `seq_0017` 使用了截断范围。为保证严格同帧比较，本报告重新运行了完整生产对照，结果保存在 `artifacts/lost_experiments/full/control_production/`。下文相对变化全部以这组完整对照为准。

### 状态策略

本文把决定状态的条件统一称为阈值，包括 uncertain 阈值和 LOST 阈值。

| 名称 | 实验行为 |
| --- | --- |
| `control_production` | 当前生产状态机；只有 TRACKING 和 UNCERTAIN，不启用 LOST 搜索。 |
| `immediate_q80` | LOST 阈值取最近 10 个稳定分数中第 8 大；上一帧满足条件，下一帧进入 LOST；一次触发。 |
| `rollback_q90` | LOST 阈值取第 9 大；连续两次低于阈值后触发；回到第一次触发帧，以 LOST 重跑目标帧和当前帧，之后暂停 LOST 计数两帧。 |
| `hysteresis_q90` | 外部模式启发的第 4 方案；最近 3 帧中 2 帧低于第 9 大阈值进入 LOST；LOST 后连续 2 帧达到 uncertain 阈值且测量被接受才退出。 |

### LOST ViewSpec

| 名称 | 实验行为 |
| --- | --- |
| `cube6_type1` | 第一轮以预测中心为中心生成 cubemap 6 视图；以第一轮 Fusor 中心为第二轮中心，使用固定 `120` 度 FOV 的 Type1 四视图。 |
| `dual_cube12` | 单轮使用两组互补朝向 cubemap，共 12 视图；Fusor 结果直接作为输出。 |
| `cube6_adaptive_type1` | 第一轮 cubemap 6 视图；第二轮以 Fusor 中心和候选尺寸动态选择 FOV，再生成 Type1 四视图。 |

## 聚合结果

括号中的 IoU 是相对生产对照的绝对变化；loss rate 和 absent FPR 括号中为百分点变化。loss rate 指可见帧 circular ERP IoU 为零的比例，不等于状态机处于 LOST 的比例。

| 状态策略 | LOST ViewSpec | Mean IoU | Success@0.5 | Loss rate | 零 IoU 帧 | Spherical IoU | Absent FPR | P95 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 对照 | 当前 `4+4` | 0.284803 | 22.23% | 21.03% | 701 | 0.241497 | 56.07% | 342.05 |
| Q80 即时 | cube6 + 固定 Type1 | 0.283045 (-0.001759) | 21.18% | 19.29% (-1.74) | 643 | 0.248912 | 57.80% (+1.73) | 570.92 |
| Q80 即时 | dual cube12 | 0.292396 (+0.007593) | 22.05% | 16.14% (-4.89) | 538 | 0.260868 | 61.27% (+5.20) | 621.98 |
| **Q80 即时** | **cube6 + 自适应 Type1** | **0.293914 (+0.009111)** | **22.56%** | **16.08% (-4.95)** | **536** | **0.260470** | **50.29% (-5.78)** | **528.51** |
| Q90 回溯 | cube6 + 固定 Type1 | 0.282309 (-0.002494) | 21.75% | 19.92% (-1.11) | 664 | 0.241115 | 57.80% (+1.73) | 870.85 |
| Q90 回溯 | dual cube12 | 0.293106 (+0.008302) | **23.52%** | 17.52% (-3.51) | 584 | 0.253575 | 57.23% (+1.16) | 851.49 |
| Q90 回溯 | cube6 + 自适应 Type1 | **0.294763 (+0.009960)** | 23.10% | 16.74% (-4.29) | 558 | 0.252747 | 54.34% (-1.73) | 878.33 |
| Q90 迟滞 | cube6 + 固定 Type1 | 0.278882 (-0.005921) | 21.24% | 19.02% (-2.01) | 634 | 0.238239 | 54.91% (-1.16) | 472.57 |
| Q90 迟滞 | dual cube12 | 0.272821 (-0.011983) | 19.83% | 18.99% (-2.04) | 633 | 0.233849 | **31.79% (-24.28)** | 554.59 |
| Q90 迟滞 | cube6 + 自适应 Type1 | 0.286406 (+0.001603) | 22.26% | 17.64% (-3.39) | 588 | 0.246923 | 51.45% (-4.62) | 491.18 |

## LOST 对应帧

“LOST 搜索帧”按该帧实际使用 LOST planner 统计，不按最终公开状态统计。LOST zero rate 和 success@0.5 的分母只包括目标可见的 LOST 搜索帧。

| 状态策略 | ViewSpec | LOST 搜索帧 | 搜索比例 | 可见 LOST 帧 | LOST mean IoU | LOST zero rate | LOST success@0.5 | 回溯事件 | 重跑帧比例 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q80 即时 | cube6 + 固定 Type1 | 991 | 28.27% | 922 | 0.265277 | 19.63% | 17.68% | 0 | 0% |
| Q80 即时 | dual cube12 | 999 | 28.49% | 946 | 0.257429 | **15.86%** | 14.90% | 0 | 0% |
| Q80 即时 | cube6 + 自适应 Type1 | 984 | 28.07% | 923 | **0.266793** | 17.88% | **17.88%** | 0 | 0% |
| Q90 回溯 | cube6 + 固定 Type1 | 261 | 7.44% | 245 | 0.236706 | 22.04% | 10.61% | 261 | 14.89% |
| Q90 回溯 | dual cube12 | 242 | 6.90% | 227 | **0.263974** | **12.33%** | **16.30%** | 242 | 13.80% |
| Q90 回溯 | cube6 + 自适应 Type1 | 259 | 7.39% | 241 | 0.243599 | 19.09% | 12.45% | 259 | 14.77% |
| Q90 迟滞 | cube6 + 固定 Type1 | 1220 | 34.80% | 1121 | 0.184035 | 31.94% | 11.95% | 0 | 0% |
| Q90 迟滞 | dual cube12 | 1319 | 37.62% | 1201 | 0.149845 | 35.64% | 8.41% | 0 | 0% |
| Q90 迟滞 | cube6 + 自适应 Type1 | 1209 | 34.48% | 1101 | 0.186392 | 32.79% | 13.90% | 0 | 0% |

回溯的“重跑帧比例”是额外重跑帧执行数除以 `3506` 个逻辑帧。每次回溯重跑第一次触发帧和当前帧；本次目标帧没有重叠，因此它也等于唯一受回溯影响帧的比例。

## 性能成本

| 状态策略 | ViewSpec | 平均 views/帧 | 平均 forwards/帧 | P50 ms | P95 ms | P99 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 对照 | 当前 `4+4` | 8.000 | 2.000 | 263.93 | 342.05 | 378.51 |
| Q80 即时 | cube6 + 固定 Type1 | 8.565 | 2.000 | 409.28 | 570.92 | 649.47 |
| Q80 即时 | dual cube12 | 9.140 | 1.715 | 372.44 | 621.98 | 732.13 |
| Q80 即时 | cube6 + 自适应 Type1 | 8.561 | 2.000 | 402.91 | 528.51 | 594.37 |
| Q90 回溯 | cube6 + 固定 Type1 | 9.340 | 2.298 | 396.37 | 870.85 | 984.13 |
| Q90 回溯 | dual cube12 | 9.380 | 2.207 | 348.09 | 851.49 | 1060.51 |
| Q90 回溯 | cube6 + 自适应 Type1 | 9.330 | 2.295 | 412.40 | 878.33 | 993.06 |
| Q90 迟滞 | cube6 + 固定 Type1 | 8.696 | 2.000 | 369.21 | 472.57 | 530.98 |
| Q90 迟滞 | dual cube12 | 9.505 | 1.624 | 377.78 | 554.59 | 601.36 |
| Q90 迟滞 | cube6 + 自适应 Type1 | 8.690 | 2.000 | 381.64 | 491.18 | 547.05 |

dual cube12 的平均 forward 数低于 2，是因为 LOST 帧只执行一次 12-view forward；这不代表延迟更低。其大 batch 和长尾使 P95/P99 仍明显高于对照。

## 关键逐序列结果

下表比较生产对照和聚合最均衡的 `immediate_q80 + cube6_adaptive_type1`。

| 序列 | 对照 mean IoU | 候选 mean IoU | 对照 loss rate | 候选 loss rate | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| `train_real/seq_0005` | 0.318654 | 0.314500 | 0.36% | 0.36% | 无 loss 收益，IoU 小幅下降。 |
| `train_sim/seq_0010` | 0.454203 | 0.435634 | 0% | 0% | 无 loss 收益，IoU 下降。 |
| `train_sim/seq_0017` | 0.269560 | 0.171812 | 32.15% | 53.65% | 严重回归，新增 103 个零 IoU 帧。 |
| `train_sim/seq_0075` | 0.339973 | 0.368601 | 15.81% | 12.03% | 明确改善。 |
| `train_sim/seq_0045` | 0.176281 | 0.244138 | 42.16% | 19.79% | 主要收益来源，减少 251 个零 IoU 帧；absent FPR 同时降低 5.78 个百分点。 |

`seq_0017` 上所有九个候选的 loss rate 都高于对照，最佳也只是 `rollback_q90 + cube6_adaptive_type1` 的 `35.28%`，仍高于对照 `32.15%`。这说明当前问题不只是某个阈值或 ViewSpec 参数，而是“切换到全局 LOST 搜索后覆盖/候选提交破坏了仍可工作的局部跟踪”。

## 方案判断

### 状态策略

1. `immediate_q80`：聚合 loss rate 最好，但进入 LOST 过于频繁，三个 ViewSpec 都约有 `28%` 的逻辑帧使用 LOST planner。它能修复 `seq_0045`，也会在 `seq_0017` 上过早放弃局部跟踪。
2. `rollback_q90`：触发较少，`seq_0017` 回归相对最小，但回溯覆盖约 `14%` 至 `15%` 的帧执行，P95 增至 `851` 至 `878 ms`。相对 Q80 自适应方案，多出的最高 mean IoU 只有 `0.000849`，不足以支付复杂度和延迟。
3. `hysteresis_q90`：退出条件过严，使系统在 LOST 中停留过久。低 absent FPR 与低 LOST IoU 同时出现，说明它更像在抑制输出，而不是可靠找回。

### ViewSpec

1. `cube6_adaptive_type1`：在三个状态策略中都取得各自最高或接近最高的整体 IoU/loss 组合，且 absent FPR 均优于对应 fixed Type1。保留。
2. `dual_cube12`：Q80 下 loss rate 较好，但 absent FPR 恶化到 `61.27%`；回溯下 success@0.5 较高但长尾延迟最大；迟滞下准确率明显下降。淘汰当前实现。
3. `cube6_type1`：固定第二轮 FOV 缺乏候选尺度适配，聚合表现普遍弱于 adaptive。淘汰。

## 第 4 策略的外部依据

- [PyTracking/DiMP 参数](https://github.com/visionml/pytracking/blob/master/pytracking/parameter/dimp/dimp50.py) 区分 `target_not_found_threshold`、`hard_negative_threshold` 等不同语义阈值，支持“不用单一阈值承担所有状态判断”。
- [Deep SORT Track](https://github.com/nwojke/deep_sort/blob/master/deep_sort/track.py) 使用连续命中 `n_init` 确认轨迹，并用连续漏检上限 `max_age` 决定删除，支持状态变化需要时间确认。
- [ByteTrack tracker](https://github.com/FoundationVision/ByteTrack/blob/main/yolox/tracker/byte_tracker.py) 保留 `lost_stracks`、`max_time_lost` 和 `re_activate`，支持 LOST 缓冲与再激活，而不是一次低分立即永久切换。

这些模式支持本次第 4 策略采用时间窗口与连续恢复确认，但实验表明当前 `2-of-3` 进入加连续两帧退出仍不适合本项目：它没有解决候选质量和恢复提交问题，只延长了 LOST 停留时间。

## 下一轮建议

下一轮不应继续只调第 8/第 9 大阈值。建议建立一个新的隔离候选：

1. 使用 `cube6_adaptive_type1` 作为唯一 LOST ViewSpec。
2. 采用 Q90 连续两次低分触发，但不做回溯，不覆盖已经提交的历史帧。
3. LOST 搜索先作为 shadow proposal；只有恢复候选连续两帧被接受，并同时满足外观分数、运动一致性和尺度跳变限制，才替换公开输出并退出 LOST。
4. 在确认恢复前保留原局部 tracking proposal 和运动预测作为并行候选，不因一次 LOST 搜索失败立即丢弃局部链路。
5. 把 `seq_0017` 设为硬性回归门槛：mean IoU 和 loss rate 均不得差于生产对照；同时要求总体 loss rate、absent FPR、P95/P99 不恶化。

## 产物

- 九组矩阵：`artifacts/lost_experiments/full/matrix.json`
- 完整生产对照：`artifacts/lost_experiments/full/control_production/aggregate.json`
- 各组合聚合：`artifacts/lost_experiments/full/<policy>__<view>/aggregate.json`
- 逐序列结果：`artifacts/lost_experiments/full/<policy>__<view>/evaluations/*.json`
- 逐候选数据：`*.candidates.jsonl`
- 逐帧 timing：`*.timings.jsonl`

本报告只记录离线测试结果和后续建议，不代表任何 LOST 方案已经进入生产实现。
