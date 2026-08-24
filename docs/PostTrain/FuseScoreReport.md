# FuseScore / IoU-BBox Refinement 实验报告

日期：2026-08-24
分支：`feat/Effeciency`（已包含 main 的 `PostTrainV2.3` 合并提交 `3982883`）

## 1. 结论摘要

本轮按要求完成四组方案、每组 3 条 validation seq 的端到端实验，并与
`E:\tringData\shared_control\production2` 中的 V2 报告对比。没有读取 holdout。

当前不建议把任何方案直接打开为生产默认：

- **方案一 delta refinement** 是最值得继续优化的候选，但在正式 gate 后三条序列的候选级 ERP 改善并不稳定，`seq_0045` absent FPR 仍恶化。
- **方案二 quality-aware refinement** 在 `seq_0017` IoU 提升，但真实 `seq_0036` 和 success@0.5 回归，暂不保留。
- **方案三 tracking scale clamp** 对定位和尺寸稳定性有效，但 `seq_0045` absent FPR 从 50.29% 升至 59.54%，只能作为“有 presence gate 的条件候选”。
- **方案四 temporal shape prior** 在 `seq_0017` 发生明显自反馈回归，不建议当前实现继续使用。

## 2. 方案与隔离

1. `iou_refine_head`：HiT embedding、初始 bbox 和 corner stability 预测 `dx/dy/log-w/log-h`，局部框裁剪后按生产 Geometry 回投。
2. `iou_refine_quality_aware`：在方案一基础上加入 corner heatmap 的位置/方差/峰值统计特征。
3. `tracking_scale_clamp`：仅当状态为 `TRACKING` 时，将每次接受框的 width/height 分别限制为上一稳定框的 `0.7~1.3`；`UNCERTAIN` 不限制，中心、视图数、候选排序、模板和状态机逻辑不主动改变。
4. `fuse_temporal_shape_prior`：以候选与上一稳定框的 width、height、aspect ratio 的 log-space 误差计算相似度，按
   `confidence + weight * similarity * (1-confidence)` 加权，不使用简单面积。

网络设计参考了 IoU-aware detector、Varifocal/Quality Focal、BorderDet 和通用 BBRefinement 的共同思路：将定位质量作为独立信号，并使用 corner/border 证据辅助修正。

## 3. 数据、训练和校准

- 训练：`E:\NewDownload\train\manifest.jsonl` 的 `train` split。
- 最终比较：`validation` split。
- 禁止：`holdout`。
- refinement 两个 checkpoint 独立保存：
  - `E:\InstaTargetingSystemTraining\checkpoints\fuse_refine_delta\best.pth`
  - `E:\InstaTargetingSystemTraining\checkpoints\fuse_refine_quality_aware\best.pth`
- 两个 checkpoint 均为 **100-step pilot**，validation 选择阶段限制为 200 batch；不能视为已完成生产训练。
- 校准使用独立 `calibration` split，6 条序列、每条 20 帧、每组 480 candidates；artifact 位于 `artifacts/FuseScore/*/calibration.json`。

## 4. V2 基线

| seq | ERP IoU | spherical IoU | success@0.5 | loss | absent FPR | P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| train_sim/seq_0017 | 0.234073 | 0.231972 | 0.229645 | 0.413361 | 0.000000 | 99.81 |
| train_sim/seq_0045 | 0.192661 | 0.188570 | 0.163993 | 0.346702 | 0.502890 | 97.54 |
| train_real/seq_0036 | 0.317540 | 0.284981 | 0.320803 | 0.341916 | 0.000000 | 96.80 |

## 5. 最终结果

### 5.1 方案一：iou_refine_head

| seq | ERP IoU | Δ | spherical IoU | success@0.5 | loss | absent FPR | trigger / fallback | candidate local Δ / ERP Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| seq_0017 | 0.229983 | -0.004090 | 0.225414 | 0.233820 | 0.407098 | 0.000000 | 100% / 0% | -0.002055 / -0.003789 |
| seq_0045 | 0.204457 | +0.011796 | 0.199382 | 0.178253 | 0.353832 | 0.450867 | 100% / 2.94% | -0.007198 / -0.000606 |
| seq_0036 real | 0.302493 | -0.015047 | 0.273878 | 0.317199 | 0.387230 | 0.000000 | 100% / 0.33% | -0.007294 / +0.001849 |

候选框本身没有稳定 local IoU 增益；端到端收益主要来自状态轨迹变化，不能把它解释为“refinement head 已经提升框质量”。

### 5.2 方案二：iou_refine_quality_aware

| seq | ERP IoU | Δ | spherical IoU | success@0.5 | loss | absent FPR | trigger / fallback | candidate local Δ / ERP Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| seq_0017 | 0.256268 | +0.022195 | 0.245561 | 0.217119 | 0.323591 | 0.000000 | 100% / 0% | -0.011872 / -0.009976 |
| seq_0045 | 0.214732 | +0.022071 | 0.207679 | 0.169340 | 0.279857 | 0.369942 | 100% / 10.50% | +0.001761 / -0.004337 |
| seq_0036 real | 0.291391 | -0.026149 | 0.269047 | 0.289907 | 0.394954 | 0.000000 | 100% / 2.41% | -0.004588 / +0.005533 |

`seq_0017` 的 IoU 提升伴随 success@0.5 下降 1.25 pp；真实 seq 回归 3.09 pp success、loss 恶化 5.30 pp，暂不接受。

### 5.3 方案三：tracking_scale_clamp

| seq | ERP IoU | Δ | spherical IoU | success@0.5 | loss | absent FPR | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| seq_0017 | 0.252905 | +0.018831 | 0.238944 | 0.237996 | 0.356994 | 0.000000 | 88.69 |
| seq_0045 | 0.229699 | +0.037038 | 0.223221 | 0.185383 | 0.224599 | 0.595376 | 96.13 |
| seq_0036 real | 0.291382 | -0.026158 | 0.263288 | 0.298661 | 0.416581 | 0.000000 | 94.41 |

尺寸误差 P95 在 `seq_0017/0045` 明显下降，但 absent FPR 在 `seq_0045` 恶化 9.25 pp。下一版必须增加 presence/absent gate，不能只靠尺寸 clamp。

### 5.4 方案四：fuse_temporal_shape_prior

| seq | ERP IoU | Δ | spherical IoU | success@0.5 | loss | absent FPR | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| seq_0017 | 0.189728 | -0.044345 | 0.182780 | 0.179541 | 0.486430 | 0.000000 | 80.40 |
| seq_0045 | 0.212304 | +0.019643 | 0.207553 | 0.176471 | 0.272727 | 0.439306 | 103.45 |
| seq_0036 real | 0.308145 | -0.009396 | 0.278501 | 0.311020 | 0.361998 | 0.000000 | 89.17 |

`seq_0017` 的明显回归说明 shape bonus 会在错误框被提交后形成自增强；当前实现不应进入生产。

## 6. 专用记录与门槛判断

已写入每个 `report.candidates.jsonl`：

- initial/refined local bbox；initial/refined ERP bbox；local IoU 和 ERP IoU；
- refinement trigger/fallback；local/ERP 改善；
- 端到端 summary 的 trigger rate、fallback rate、improvement 均值。

目前没有把 `tracking_scale_clamp` 的触发次数单独注入 summary counter；报告只使用最终指标，避免伪造未采集的计数。

预设门槛：success@0.5 不下降超过 0.5 pp，loss/FPR 不恶化超过 1 pp，`seq_0017` 不回归，`seq_0045` absent 不出现未解释错误。四组均至少有一项未通过，因此没有自动改生产默认开关。

## 7. 推荐后续方案

推荐优先做 **方案一 v2（geometry-aware acceptance）**：

1. 保持 delta head，但接受条件同时检查 local box 合法性、回投 ERP IoU proxy、corner stability 和 predicted IoU；
2. 若 quality head/corner stability 低于 calibration 分位数，回退初始框；
3. 对 `TRACKING` 使用 0.7–1.3 clamp，但在 absent/presence 低于阈值时禁止 clamp 结果提交；
4. temporal shape prior 只作为 tie-break（权重不超过 0.02），且 reference 必须来自连续两帧稳定框，避免单帧自反馈；
5. 重新进行完整 train/validation/calibration 训练，pilot 结果不得直接晋级。

生产默认保持 V2；所有方案仍保留在 `artifacts/FuseScore` 供复核。
