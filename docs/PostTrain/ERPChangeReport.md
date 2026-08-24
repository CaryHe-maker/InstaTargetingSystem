# ERP 第二轮直接裁图实验报告

## 1. 结论

本轮 ERP change 没有产生可进入生产候选的变体。

- `erp_crop_2x_strict`：回退。5 条预筛序列完整，但总体精度下降，`train_sim/seq_0017` 出现不可接受的硬回归。
- `erp_crop_2x_relaxed`：回退。Success@0.5 和 `seq_0045`、`seq_0078` 有明显收益，但总体 loss rate 上升，`seq_0017` 和 `train_real/seq_0005` 严重退化。
- `erp_crop_4x_relaxed`：仅保留为单序列诊断结果。它在 `seq_0045` 上优于生产基线，但 absent FPR 明显差于 `2x_relaxed`，且同时改变了第一轮 FOV，不能把收益归因于 4x crop。
- `erp_crop_2x_3x_best`：未运行。现有 2x 路径尚未保护 `seq_0017`，此时增加 3x 分支和候选选择没有继续验证价值。

因此实验 2.1 的决定是保持共享生产路径：第二轮继续使用围绕第一轮 Fusor 中心生成的 Geometry Type1 透视视图，不启用 ERP 原图直接 crop。

## 2. 数据范围和完整性

### 2.1 5-seq 预筛

`erp_crop_2x_strict` 和 `erp_crop_2x_relaxed` 使用以下 validation 序列：

```text
train_sim/seq_0045
train_sim/seq_0017
train_real/seq_0005
train_sim/seq_0078
train_sim/seq_0010
```

运行产物位于 `E:\tringData\erp_5seq_v1`，共享对照位于 `E:\tringData\shared_control\production`。

- 计划任务：10。
- 完成任务：10。
- 失败任务：0。
- 每个变体完成 5/5 序列，共对齐 3333 个 visible 帧和 173 个 absent 帧。
- runner 已检查 report、candidate、timing、有限值和 `_SUCCESS.json` artifact hash。
- Git commit：`692a7747bdb0d37ff78910189102d823da87f591`，运行时工作树为 dirty。
- manifest SHA256：`611b4dbf1dcf47aaab531aacb91b12b83b4196029f0c839ccc722d23e33f2982`。
- checkpoint SHA256：`23f7e6e5981eb29e2f4bc8027f2728a4600438efc7a61daefdc8587b492db73c`。
- config SHA256：`139764dfb048308a14ad41ac0303942d90b7c8d05d1f1eb2729826bc4e429be0`。
- 未读取 holdout。

这是 5-seq 预筛，不是 `TryingPlan.md` 规定的完整 15-seq validation。两个 2x 变体已经在硬回归序列上失败，因此没有必要用额外 10 条序列扩大失败实现的运行成本。

### 2.2 4x 单序列诊断

`erp_crop_4x_relaxed` 只运行了 `train_sim/seq_0045`，完整处理 1296 帧，其中 1122 个 visible 帧、173 个 absent 帧。报告位于：

```text
E:\tringData\erp_crop_4x_relaxed_seq0045_no_visual\report.json
```

该变体当前语义为：

- 使用 `2.0x` 框判断第二轮 ERP 路径是否可执行；
- 判断通过后实际裁取 `4.0x` 框，超出 ERP 边缘的部分裁去；
- TRACKING 和 UNCERTAIN 第一轮 Type1 均固定为 `120° x 120°`；
- 第二轮直接 ERP 图仍缩放到 HiT 输入尺寸。

因为它同时改变 crop 尺度、触发条件和第一轮 FOV，所以只作为组合诊断，不能与 2x 变体作严格单变量归因。

## 3. 5-seq 聚合结果

### 3.1 Macro：逐序列等权平均

| 变体 | Mean IoU | Spherical IoU | Success@0.5 | Loss rate | 宽度相对误差 | 高度相对误差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `shared_control_production` | 0.326417 | 0.272208 | 28.13% | 15.25% | 1.146101 | 0.760729 |
| `erp_crop_2x_strict` | 0.277825 | 0.220264 | 25.24% | 31.45% | 0.859261 | 0.577487 |
| `erp_crop_2x_relaxed` | 0.327600 | 0.270956 | 33.50% | 28.78% | 0.762432 | 0.630652 |

相对共享对照：

- `2x_strict`：Mean IoU `-0.048592`，Spherical IoU `-0.051944`，Success@0.5 `-2.89` 个百分点，loss rate `+16.20` 个百分点。
- `2x_relaxed`：Mean IoU `+0.001182`，Spherical IoU `-0.001252`，Success@0.5 `+5.37` 个百分点，但 loss rate `+13.53` 个百分点。

`2x_relaxed` 的 macro Mean IoU 基本持平不能掩盖 loss rate 和硬回归序列的恶化。

### 3.2 Micro：全部逐帧合并

| 变体 | Visible 帧 | Mean IoU | Spherical IoU | Success@0.5 | 零 IoU/loss | pooled Absent FPR | pooled P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `shared_control_production` | 3333 | 0.294693 | 0.248449 | 24.09% | 19.11% | 56.07% | 314.23 ms |
| `erp_crop_2x_strict` | 3333 | 0.269940 | 0.219067 | 24.42% | 31.80% | 0.00% | 353.24 ms |
| `erp_crop_2x_relaxed` | 3333 | 0.299470 | 0.253214 | 30.27% | 32.58% | 4.62% | 334.04 ms |

两种 ERP 路径都显著降低了 `seq_0045` 的 absent FPR，但同时增加 visible 帧上的零 IoU。当前提交和 valid 语义把 absent 抑制收益与 visible tracking loss 绑定在一起，不能只根据 absent FPR 选择方案。

## 4. 逐序列结果

### 4.1 Mean IoU / loss rate

| 序列 | 对照 Mean IoU | strict Mean IoU | relaxed Mean IoU | 对照 loss | strict loss | relaxed loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `train_sim/seq_0045` | 0.176281 | 0.181154 | 0.228997 | 42.16% | 47.77% | 46.97% |
| `train_sim/seq_0017` | 0.269560 | 0.035852 | 0.101052 | 32.15% | 85.39% | 69.73% |
| `train_real/seq_0005` | 0.318654 | 0.397407 | 0.274089 | 0.36% | 1.80% | 26.74% |
| `train_sim/seq_0078` | 0.413389 | 0.320510 | 0.579658 | 1.56% | 22.27% | 0.45% |
| `train_sim/seq_0010` | 0.454203 | 0.454203 | 0.454203 | 0.00% | 0.00% | 0.00% |

### 4.2 关键分层判断

#### `train_sim/seq_0017`

这是计划中规定的局部跟踪链硬门槛。两个 2x 变体都失败：

- strict：Mean IoU `-0.233708`，loss rate `+53.24` 个百分点；
- relaxed：Mean IoU `-0.168508`，loss rate `+37.58` 个百分点。

该序列 ERP 直接裁图触发率为 strict `99.16%`、relaxed `94.57%`。直接路径实际上接近替换了整个第二轮，而不是只覆盖少量安全帧。这与严重回归同时出现，足以触发回退规则。

#### `train_sim/seq_0045`

该序列包含 173 个 absent 帧：

- 生产对照 absent FPR 为 `56.07%`；
- strict 降至 `0.00%`；
- relaxed 降至 `4.62%`。

但 strict/relaxed 的 visible loss rate 分别从 `42.16%` 上升到 `47.77%` 和 `46.97%`。Absent 抑制是真实收益，但没有同时保护 visible tracking。

#### `train_real/seq_0005`

strict 的 Mean IoU 提升到 `0.397407`，但 relaxed 的 loss rate 从 `0.36%` 升到 `26.74%`。relaxed 在该序列的直接触发率达到 `98.80%`，说明 `1.25x` 几何判断过于宽松，无法充当跟踪稳定性判断。

#### `train_sim/seq_0078`

relaxed 是本轮收益最大的分层：Mean IoU 从 `0.413389` 提升到 `0.579658`，Success@0.5 从 `41.43%` 提升到 `70.60%`，loss rate略降。与此同时 strict 明显退化，证明结果对触发规则和轨迹状态非常敏感，不能把 ERP direct crop 视为稳定的统一优化。

#### `train_sim/seq_0010`

两个变体与对照结果完全一致，ERP direct crop 触发率均为 `0%`。但 P95 仍从对照 `246.24 ms` 增至 strict `360.94 ms`、relaxed `348.32 ms`。由于结果路径未变化，这部分性能差异更可能是单次运行波动、环境状态或计时噪声；本轮性能数据不能支持确定性的加速结论。

## 5. ERP 路径触发率

| 变体 | 范围 | direct ERP 帧 | 可处理帧 | 触发率 |
| --- | --- | ---: | ---: | ---: |
| `erp_crop_2x_strict` | 5-seq | 2423 | 3506 | 69.11% |
| `erp_crop_2x_relaxed` | 5-seq | 2618 | 3506 | 74.67% |
| `erp_crop_4x_relaxed` | `seq_0045` only | 912 | 1295 | 70.42% |

按序列观察，触发率从 `0%` 到接近 `100%`。仅根据 crop 是否越界进行分支判断，无法控制该路径对不同运动、尺度和跟踪稳定性序列的影响。

## 6. 4x 单序列结果

| 变体 | Mean IoU | Spherical IoU | Success@0.5 | Loss rate | Absent FPR | P50 | P95 | P99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `shared_control_production` | 0.176281 | 0.169567 | 14.62% | 42.16% | 56.07% | 243.00 ms | 322.24 ms | 381.56 ms |
| `erp_crop_2x_relaxed` | 0.228997 | 0.236136 | 23.44% | 46.97% | 4.62% | 221.61 ms | 362.48 ms | 400.66 ms |
| `erp_crop_4x_relaxed` | 0.206975 | 0.205029 | 20.05% | 35.12% | 33.53% | 195.57 ms | 334.98 ms | 375.89 ms |

相对生产对照，4x 组合在 `seq_0045` 上：

- Mean IoU `+0.030694`；
- Spherical IoU `+0.035462`；
- Success@0.5 `+5.44` 个百分点；
- loss rate `-7.04` 个百分点；
- absent FPR `-22.54` 个百分点；
- P50 `-47.43 ms`，P95 `+12.74 ms`，P99 `-5.68 ms`。

相对 `2x_relaxed`，4x 的 loss rate 和延迟更好，但 Mean IoU、Spherical IoU、Success@0.5 和 absent FPR 都更差。尤其 absent FPR 从 `4.62%` 回升到 `33.53%`。

该结果不足以支持 4x 晋级，原因有三：

1. 只有一条序列，没有运行 `seq_0017` 硬门槛；
2. 同时把第一轮 TRACKING/UNCERTAIN FOV 固定为 120°，不是 crop 尺度的单变量实验；
3. 4x crop 曾暴露超过半球时 BFoV span 大于 `pi` 的回投边界，虽然已经加入上限处理并完成重跑，但仍需要 seam、宽框和极点回归。

## 7. 决定与后续建议

### 7.1 本轮决定

- 保持 `shared_control_production` 为生产默认。
- 回退 `erp_crop_2x_strict`。
- 回退 `erp_crop_2x_relaxed`。
- `erp_crop_4x_relaxed` 不具备比较资格，不进入候选。
- 暂不运行 `erp_crop_2x_3x_best`。

### 7.2 若继续研究 ERP direct crop

下一次实验应先拆分变量，而不是继续扩大 crop 倍数：

1. `4x crop + 动态第一轮 FOV`，隔离 4x crop 本身；
2. `2x crop + 第一轮固定 120°`，隔离固定大视场的影响；
3. 所有新变体先只运行 `seq_0017`，不满足 Mean IoU 和 loss rate 硬门槛立即停止；
4. 通过 `seq_0017` 后再运行 `seq_0045` 检查 absent FPR；
5. 分支 gate 不应只检查 ERP 边缘，还应使用因果可观测的稳定性条件，例如近期已接受框的稳定程度、状态、尺度创新量和 crop 内目标占比；
6. 性能结论至少重复运行多次，并在 direct trigger 为 0 的序列上验证路径关闭时没有额外开销。

在上述保护完成前，不建议继续全 15-seq 或 holdout 测试。
