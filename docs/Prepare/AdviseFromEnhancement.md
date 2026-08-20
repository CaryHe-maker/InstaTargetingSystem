# Stage 3 后 IoU 优先优化与推理效率实施方案

> 状态：阶段 0 至阶段 4 已执行；阶段 5 未开始。本文同时保留后续实验蓝本。
>
> 优先级：先提高定位 IoU，再在不破坏定位质量、目标存在判断和事务语义的前提下提高效率。
>
> 适用范围：`enhancement/preTraining` 在 Stage 3 训练完成后的 calibration、validation、推理优化和最终验收。本文不授权修改正在运行的 Stage 3、训练 manifest、训练依赖、checkpoint 或 final holdout。

## 当前执行结果（2026-08-20）

- Stage 3 最优权重已冻结到 `models/hit_small_stage3.pth`，SHA-256 为 `23f7e6e5981eb29e2f4bc8027f2728a4600438efc7a61daefdc8587b492db73c`；6000 step，490 个模型张量均有限。
- E02 使用 `E:\NewDownload\train\manifest.jsonl` 的 calibration split 拟合 Stage 3 `presence*predictedIoU`。4792 个候选的 Brier `0.17895 -> 0.13769`、ECE `0.18271 -> 0.01685`。
- A1 比较 presence、quality 与乘积后选择乘积；SingleScore 选择 appearance/motion `0.50/0.50`，工作点为 `0.597262/0.740642`。
- Stage 4 误差分解定位到融合框形状问题。A4 `best_source` 在同一 450 帧 validation 序列上将 circular ERP mean IoU 从 `0.4415` 提升到 `0.4542`，success@0.5 从 `0.3118` 提升到 `0.4098`，spherical mean IoU 从 `0.2068` 提升到 `0.3575`，invalid BFoV 从 194 降到 0，因此成为生产配置。其他 A2/A3/A5/A6/A7 没有足够单变量证据，不写入生产路径。
- 4 个 validation 序列、1796 个可见帧的聚合 circular ERP mean IoU 为 `0.35915`，tracking loss rate 为 `0.11971`（215 帧），P95 为 `355.80 ms`。丢失率仅统计，不触发 LOST 或恢复。
- 串行 `4+4`、FP32、speculative disabled 保持不变。FP16、GPU crop、batch 8、推测生产接入和 TensorRT 均属于阶段 5，尚未开始。
- final holdout 尚未读取。

## 1. 目标、约束与决策原则

### 1.1 优化目标

本项目采用字典序目标，而不是把精度和速度简单相加：

1. 首先最大化 validation 上的 circular ERP IoU、success AUC、success@0.5 和 BFoV spherical IoU。
2. 在定位指标不退化、目标缺失误报率不恶化的候选中，最小化端到端 P95，其次最小化 P99、峰值显存和每帧 forward 数。
3. final holdout 只用于一次冻结后的最终报告，不参与模型、阈值、回退条件或优化后端选择。

任何只提高平均 FPS、但降低 IoU 或恶化 P95/P99 的方案均不满足本项目优先级。任何只提高 valid rate、但增加目标缺失误报的方案也不得视为有效改进。

### 1.2 当前已知基线

当前已有结果只能作为非 holdout 工程基线：

| 项目 | 当前观测 | 解释 |
|---|---:|---|
| legacy validation circular ERP mean IoU | 约 `0.330` | 来自 `train_sim/seq_0010` 的 450 帧运行 |
| legacy 端到端 P95 | 约 `386.8 ms` | 必须以本系统实测为准，不能引用 HiT 单视图 FPS 替代 |
| Stage 3 + calibrated + best_source mean IoU | `0.35915` | 4 个 validation 序列、1796 个可见帧 |
| Stage 3 tracking loss rate | `0.11971` | 215 个 circular ERP IoU 为零的可见帧 |

这些数据是非 holdout 工程基线，不是最终发布结论。Stage 3 阈值和校准已由 calibration split 冻结；final holdout 仍不得用于继续选择参数。

### 1.3 不可破坏的语义

- 初始化帧仍使用 frame 0 anchor。
- 普通 `TRACKING`/`UNCERTAIN` 帧仍保留两轮 `4+4` 和统一 Fusor 候选池。
- Round 1 只决定搜索方向；Round 2 和统一候选池负责最终框。
- Round 1 不更新正式运动历史、模板、公开 bbox/BFoV、状态分数或 Controller revision。
- Geometry 的 seam-aware ERP、球面投影和 BFoV 语义不能为了速度被替换为普通平面近似。
- speculative pipeline 必须默认关闭；失效时完整回退串行 `4+4`。
- final holdout 在模型、校准、Controller 参数和运行时开关冻结前不可读取。

## 2. 实验资产和可复现性

### 2.1 固定四类数据用途

| split | 唯一用途 | 禁止用途 |
|---|---|---|
| train | Stage 3 参数学习 | 阈值验收、最终报告 |
| validation | checkpoint 选择、Geometry/Controller/运行时 A/B | 拟合概率映射、最终报告 |
| calibration | 外观概率、quality、运动概率、SingleScore 和门限拟合 | 选择 backbone、报告最终泛化结果 |
| final holdout | 所有内容冻结后的一次最终评估 | 任何调参、模型选择、回退阈值选择 |

数据划分必须按 sequence 或原始视频分组，不能让相邻帧跨 split。所有实验先记录 manifest hash，并验证 split 中没有重复视频、重复帧或同源片段泄漏。

### 2.2 每次实验必须记录的身份

每个实验目录至少保存：

```json
{
  "experimentId": "E03-stage3-calibrated-serial",
  "gitCommit": "<40-char sha>",
  "configSha256": "<sha256>",
  "checkpointSha256": "<sha256>",
  "manifestSha256": "<sha256>",
  "split": "validation",
  "sequenceIds": ["..."],
  "seed": 20260820,
  "device": "RTX 4060 Laptop GPU",
  "torchVersion": "<exact>",
  "cudaVersion": "<exact>",
  "precision": "fp32",
  "warmupFrames": 30,
  "measuredFrames": 450
}
```

还必须保存原始逐帧结果、逐候选结果、运行日志、校准参数和汇总 JSON。只保存终端中打印的均值不足以复核实验。

### 2.3 公平性能测试条件

- 使用 AC 电源，固定 Windows 电源模式和 GPU 性能模式。
- 关闭训练任务或确保性能测试与训练不共享 GPU。Stage 3 运行期间不得做正式 latency 对比。
- 每个变体至少预热 30 帧；预热帧不计入分位数。
- 相同序列、相同帧顺序、相同 checkpoint、相同 Controller 参数运行至少 3 次。
- 记录 GPU 温度、功耗、利用率、峰值显存和是否发生 thermal throttling。
- CUDA forward 用 CUDA Event 测量；端到端仍用单调 CPU 时钟测量。计时边界必须在变体间一致。

## 3. 统一指标和选择规则

### 3.1 定位指标

主指标：

- `circularErpMeanIoU`：ERP 经线循环语义下的逐帧 IoU 均值。
- `successAUC`：0 到 1 IoU 阈值上的成功曲线面积。
- `successRateAt0.5`：IoU 大于 0.5 的帧比例。
- `sphericalMeanIoU`：BFoV 在球面面积权重下的 IoU。

诊断指标：

- 球面中心角误差 P50/P95。
- BFoV 宽度和高度相对误差 P50/P95。
- 直接 ERP bbox IoU 与 BFoV 间接回投 bbox IoU。
- `envelopeInflation`，并按纬度、FOV 和局部框 `normalizedRadius` 分组。
- seam、高纬度、局部视图边缘、小目标、快速运动、遮挡和 off-view negative 分层结果。

### 3.2 存在和置信度指标

- presence PR-AUC 和 ROC-AUC。
- Brier score、NLL、ECE 和 reliability diagram。
- predicted IoU 的 MAE、Spearman 相关性和分桶校准误差。
- 目标缺失误报率、目标可见漏报率、valid rate。
- 候选 `singleScore` 对 `IoU >= 0.5` 的排序 AUC。

### 3.3 效率指标

- 端到端 P50/P95/P99。
- crop、preprocess、CPU 到 GPU copy、HiT forward、projection、calibration、Controller 各阶段 P50/P95/P99。
- 每帧视图数、batch size、forward 数。
- GPU 利用率、峰值显存、OOM 次数和温度。

### 3.4 候选选择的字典序规则

建议在 validation 上按下列顺序选择：

1. 目标缺失误报率不得高于串行 Stage 3 基线。
2. circular ERP mean IoU 必须高于或等于基线；IoU 优化实验建议至少取得 `+0.01` 绝对提升，才值得增加长期复杂度。
3. success@0.5 不得下降，spherical mean IoU 不得下降超过 `0.005` 绝对值。
4. 定位约束相同的候选中选择 P95 更低者；P95 接近时选择 P99 和峰值显存更低者。
5. speculative pipeline 的文档上限允许 1% 到 2% IoU 相对下降，但由于本项目明确 IoU 优先，默认采用更严格的 `<= 0.5%` 相对下降门槛。只有产品侧明确接受精度换延迟时才可放宽，并单独记录决策。

数值门槛是 validation 阶段的工程门槛，不是从 holdout 反向调出的目标。

## 4. 总体执行顺序

```text
Stage 3 正常完成并保存不可变快照
 -> checkpoint 基本完整性与 validation 筛选
 -> calibration split 拟合外观/quality/运动/SingleScore
 -> 冻结第一版校准参数
 -> 串行 4+4 的 Geometry 误差分解
 -> 串行 4+4 的 Controller 单变量 A/B
 -> 冻结 IoU 优先候选
 -> FP32/FP16、crop、batch 4/8 的独立性能 A/B
 -> speculative pipeline validation
 -> TensorRT FP16 候选
 -> 冻结全部参数和开关
 -> final holdout 一次验收
```

不得同时改变 checkpoint、校准、Geometry、Controller 和运行时后端。每个实验只允许一个主要变量，否则无法归因。

## 5. 阶段 0：Stage 3 完成与快照

### 5.1 完成条件

- 训练进程正常退出，最后日志、best checkpoint 和 last checkpoint 均可读取。
- 保存训练 commit、`configs/train_stage3.yaml` hash、manifest hash、随机种子和训练依赖版本。
- 对 checkpoint 执行逐 key、shape、dtype 和有限值检查。
- 用 1 个正样本和 1 个负样本做只读 smoke inference，确认 corner、presence、quality 字段均存在且有限。

### 5.2 禁止操作

- 不在原 Stage 3 目录覆盖 checkpoint。
- 不修改 resume 配置后继续写入同一实验目录。
- 不因为单个训练 loss 更低直接选 checkpoint；必须使用固定 validation 指标。

## 6. 阶段 1：Stage 3 checkpoint 筛选

### 6.1 固定串行基线

使用 `speculativePipeline.enabled=false`、FP32、串行两次 batch 4，固定旧 Controller 阈值。此阶段目的是比较模型空间能力，不做阈值补偿。

对每个候选 checkpoint 运行全部 validation sequence，至少输出：

- circular ERP mean IoU、success AUC、success@0.5。
- BFoV spherical IoU、中心误差、宽高误差。
- presence/quality 原始 logit 和概率分布。
- valid rate、缺失误报、状态分布。
- P50/P95/P99，仅作辅助，不用未校准速度结果淘汰定位更好的 checkpoint。

选择规则：先按 circular ERP mean IoU，再按 spherical IoU 和 success@0.5；差异小于 bootstrap 95% 置信区间时，选择参数更稳定、缺失误报更低的 checkpoint。

### 6.2 当前工具入口

`tools/eval_manifest_controller.py` 已能运行单 sequence 并输出 circular ERP IoU、spherical IoU、中心/尺度误差、误报、valid rate 和延迟分位数，可作为实施样本：

```powershell
$env:PYTHONPATH = "src"
python tools/eval_manifest_controller.py `
  --manifest E:\NewDownload\train\manifest.jsonl `
  --config configs\RGBonly.yaml `
  --weights <stage3-checkpoint> `
  --split validation `
  --sequence <validation-sequence> `
  --output outputs\experiments\E01\<sequence>.json
```

实施前应扩展该工具以支持 sequence 列表与总汇总，并补充 success AUC、分层统计、平均视图数、forward 数和阶段耗时。`tools/eval_sequence.py` 当前是占位文件，不能作为已完成能力引用。

## 7. 阶段 2：独立 calibration

### 7.1 需要保存的原始字段

对 calibration split 每个局部候选保存：

- `presenceLogit`、`qualityLogit`、`predictedIoU`、corner score。
- 真实 visible 标签、局部预测框 IoU、ERP IoU、BFoV spherical IoU。
- `appearanceProbability`、raw/effective motion probability、motion reliability。
- 视图角色、状态、FOV、纬度、`normalizedRadius`、`edgeMargin`、`envelopeInflation`。

标签必须来自 ground truth 和 Geometry，不得使用 Controller 最终是否接受作为 presence/quality 标签。

### 7.2 外观概率拟合

推荐按以下顺序比较，使用 calibration 内部固定 K-fold 或按 sequence 分组交叉验证避免过拟合：

1. presence logit 的 temperature scaling。
2. presence 的 Beta calibration。
3. quality 输出到真实 IoU 的单调 isotonic mapping；样本不足时使用线性或 temperature mapping。
4. `p_model = calibratedPresence * calibratedQuality`。

选择时先要求 PR-AUC 不下降，再最小化 Brier/NLL/ECE。isotonic 仅在每折都有足够样本时使用；如果 reliability curve 出现大幅阶梯，应退回参数更少的方法。

### 7.3 SingleScore 和门限拟合

保持公式可解释：

```text
singleScore = w_app * p_model + w_motion * effectiveMotionProbability
w_app + w_motion = 1
```

建议网格：

- `w_app`：`0.50, 0.60, 0.70, 0.80, 0.90`。
- `candidateMinScore`：按 calibration 分数分位数生成，而不是固定等距盲扫。
- `fusionSourceMinConfidence`：同样按候选来源分数分位数生成。
- 推测方向门限：只在串行参数冻结后拟合。

目标函数采用约束选择：先限制 absent false-positive rate 不高于基线，再最大化 circular ERP IoU 和候选 hit@0.5，最后最小化 ECE。不要用 final holdout 选择网格点。

### 7.4 校准产物

校准参数应保存为独立版本化产物，至少包含模型 hash、manifest hash、fit 方法、参数、样本数、每折指标和生成 commit。不要直接覆盖 `src/instatarget/controller/fused_score.py` 中旧常量后丢失来源。

建议后续新增严格配置节或只读 calibration artifact loader，并为未知字段、hash 不匹配和非有限参数添加拒绝测试。

## 8. 阶段 3：IoU 误差分解

在改算法前，先将每个低 IoU 帧归入主要误差类型：

| 误差类型 | 判据 | 优先检查位置 |
|---|---|---|
| 中心误差 | 中心角误差高、宽高误差低 | motion、R1 搜索中心、视图覆盖 |
| 尺度误差 | 中心正确、宽高相对误差高 | corner head、quality、Fusor box mode |
| 投影包络误差 | local IoU 高、ERP IoU 低、`envelopeInflation` 高 | Geometry boundary 与 ERP envelope |
| 视图边缘误差 | `normalizedRadius` 高或 `edgeMargin` 低 | R2 中心和 FOV |
| seam 误差 | bbox 跨 0/width，经线附近 IoU 异常 | circular interval 和 fusion |
| 高纬度误差 | `abs(pitch)` 高时 spherical IoU 显著下降 | BFoV fitting、透视畸变 |
| 目标缺失误报 | visible=false 仍 valid | presence calibration、门限 |
| 遮挡后漂移 | 遮挡前后中心跳变 | motion reliability、LOST/recovery |

每类至少保存 20 个代表帧的 ERP、局部 crop、预测框、GT、候选分数和投影边界。只有当某类误差在 validation 上占比显著且可复现时，才进入代码 A/B。

## 9. 阶段 4：IoU 优先改进候选

### 9.1 A1：使用校准后的 predicted IoU 排序

目的：让 quality head 影响候选排序，但不直接改变 Geometry。

变体：

- A1-0：校准后的 presence。
- A1-1：`p_model = p_present * p_quality`。
- A1-2：`p_model = p_present * calibratedPredictedIoU`。

检查位置：`src/instatarget/tracker/pytorch_hit_session.py`、`tracker/observation.py`、`controller/fused_score.py`、`app/driver.py`。

验收：candidate ranking AUC、circular ERP IoU 和缺失误报至少三者中前两者改善，另一项不恶化。禁止直接把 predicted IoU 当作最终 Controller 置信度而跳过 presence。

### 9.2 A2：Round 1 方向估计

目的：提高 Round 2 视图对目标中心的覆盖，优先减少视图边缘框。

候选：

- 当前 Fusor 最优候选中心。
- top-k 候选按校准 `singleScore` 的球面加权中心。
- 仅对互相重叠、来源分数达标的候选做球面加权中心。

检查位置：`controller/state_evaluator.py`、`controller/fusor.py`、`controller/recovery_planner.py`。

验收：R2 目标覆盖率、R2 `normalizedRadius`、中心误差和最终 IoU。禁止让 R1 粗框更新正式尺度或运动历史。

### 9.3 A3：R2 自适应 FOV

目的：目标可靠时缩小搜索 FOV，提高局部像素分辨率；不可靠时扩大覆盖。

起始公式仅作为 validation 候选：

```text
targetFov = clamp(
    contextScale * committedAngularSize
    + uncertaintyScale * angularUncertainty,
    minFov,
    maxFov
)
```

必须分别扫描 `contextScale` 和 `uncertaintyScale`，并按目标大小、速度、纬度分层。任何缩 FOV 方案必须保持 R2 覆盖率至少 `99%`，且 off-view fallback 可用。不能以静默裁掉视图换速度。

### 9.4 A4：Fusor 框形状策略

目的：避免多个框融合后产生过大的并集包络，损失 IoU。

依次比较：

- 对照组 `REFERENCE_ADAPTIVE`。
- 最高 quality 候选框，仅用其他视图确认中心。
- 对中心和 log 宽高分别做 quality 加权的球面融合。
- 对异常尺度使用 median/M-estimator，再拟合 BFoV。

检查位置：`controller/fusor.py`、`controller/classifier.py`。

验收必须同时看中心误差和宽高误差。如果 mean IoU 上升但小目标或 seam 分层明显退化，不得全局启用，可考虑按质量/几何条件选择。

### 9.5 A5：Geometry 投影误差修正

仅当误差分解证明 `envelopeInflation` 与 IoU 损失强相关时执行：

1. 保留同一条局部框边界采样。
2. ERP bbox 使用投影边界的最短 circular horizontal interval。
3. BFoV 继续使用球面语义，不用 ERP 像素线性宽度替代。
4. 比较 direct ERP bbox、BFoV 间接回投 bbox 和当前输出。
5. 对 seam、极点和超大 FOV 增加单元测试。

检查位置：`geometry/spherical_geometry.py`、`geometry/seam.py`、`geometry/projection_math.py`。该实验不能与 Controller 权重调整同时进行。

### 9.6 A6：尺度时间滤波

只对已正式接受的 Round 2 测量进行 log 宽高滤波；滤波权重由 calibrated quality 和 scale uncertainty 决定。中心与尺度分开处理，避免高质量中心被低质量尺度拖动。

验收：宽高误差 P50/P95、IoU、快速尺度变化分层。若目标快速接近时响应滞后，则停止该方案，不用更强平滑掩盖。

### 9.7 A7：模板更新，最后考虑

当前 anchor-only 行为稳定且与 Stage 3 归因清晰。只有在前述方案完成后，且长期外观变化是主要误差来源时，才评估 recent/stable template：

- 仅在正式 Round 2 高质量提交后更新。
- 遮挡、UNCERTAIN、LOST、低 presence 或尺度跳变时禁止更新。
- anchor 永不覆盖；必须可立即回退。
- 模板实验不能与 backbone 或 calibration 同时改变。

这是高漂移风险实验，优先级低于校准、R2 覆盖和 Fusor 框形状。

## 10. 阶段 5：效率优化顺序

效率实验全部基于已经冻结的 IoU 最优串行候选。

### 10.1 P0：先补齐 profiler

当前 `src/instatarget/eval/profiler.py` 只提供 count/mean/min/max，不能满足 P50/P95/P99。应先扩展为逐样本或固定容量直方图，并记录：

- `decodeMs`、`cropMs`、`preprocessMs`、`hostToDeviceMs`。
- `inferMs`，使用 CUDA Event 并同步到正确边界。
- `projectionMs`、`calibrationMs`、`controllerMs`、`totalProcessingMs`。
- batch size、role 数、frame index、forward count。

计时本身必须做开/关 A/B，确认 instrumentation overhead 可忽略。

### 10.2 P1：PyTorch FP16

固定 checkpoint、输入尺寸和串行 `4+4`，比较 FP32 与 FP16：

- 每个局部框坐标、presence、quality 和最终 TrackResult 做数值差异统计。
- 检查 NaN/Inf；一旦出现，按现有规则回退 FP32 并记录。
- 精度门槛采用第 3.4 节，不能只看 forward time。

可独立测试 `torch.inference_mode()`、固定 shape 下的 cuDNN benchmark、channels-last。每个开关单独做实验，避免无法归因。

### 10.3 P2：数据搬运和 crop

按 profile 判断是否值得执行：

- 复用固定 shape 的 host/device buffer。
- pinned memory 与 non-blocking copy 成对测试。
- 将 crop/resize 搬到 GPU 时，先做像素级和 box 级回归，特别检查插值、坐标原点、边界填充和颜色通道。
- 不得通过降低输入尺寸或边界采样数换速度，除非另立精度实验且通过 IoU 门槛。

### 10.4 P3：batch 4/8/10

必须实测：

| 变体 | 语义 | 目的 |
|---|---|---|
| P3-0 | 两次 batch 4 | 串行基线 |
| P3-1 | 一次 batch 8 | 普通推测稳态候选 |
| P3-2 | 一次 batch 10 | LOST/恢复容量与显存边界 |
| P3-3 | 逻辑合并、物理两个 batch 4 | batch 8 OOM 或无收益时的调度 fallback |

记录 CUDA forward、端到端分位数和峰值显存。如果 batch 8 相对两个 batch 4 的 forward 节省不足 `25%`，暂停流水线复杂化，优先 GPU crop 或 FP16。

### 10.5 P4：speculative pipeline

启用顺序：

1. `enabled=false` 回归必须与串行改动前结果逐帧一致。
2. fake backend 验证 generation/revision、乱序恢复、迟到丢弃、LOST 和 sequence close。
3. validation 上只启用状态机，不合并 batch，测回退正确性。
4. 再启用 batch merge，比较一次 batch 8 与逻辑 fallback。

验收：

- 接受率目标 `>=80%`，回退率目标 `<=20%`。
- 接受帧下一轮覆盖率 `>=99%`。
- circular ERP mean IoU 相对串行下降默认不超过 `0.5%`。
- success@0.5 不下降，缺失误报不恶化。
- P95/P99 不恶化；所有 frame 仍只正式提交一次且顺序连续。

回退率是结果指标，不得通过放宽覆盖条件强行满足。

### 10.6 P5：TensorRT FP16

只有 PyTorch FP16 和 batch profile 稳定后再执行：

- 导出固定输入 shape 的 Stage 3 模型，逐输出比较 corner/presence/quality。
- 使用相同 calibration 和 Controller，不在 TensorRT 实验中重调阈值。
- 对正样本、负样本、seam、高纬度、边缘框进行数值回归。
- 保存 TensorRT、CUDA、驱动、GPU、builder flags 和 engine hash。

`tools/export_backend.py` 当前仍是占位文件，实施时需补导出、校验和错误回退。TensorRT INT8 排在最后，必须使用代表性 calibration 数据，并重新验证 spherical IoU、缺失误报和非有限输出。

## 11. 完整 A/B 矩阵

| ID | 唯一主要变量 | split | 主指标 | 进入下一阶段条件 |
|---|---|---|---|---|
| E00 | legacy 串行 FP32 | validation | 全部基线 | 产物完整 |
| E01 | Stage 3 串行，显式 identity 校准 | calibration probe | 模型原始定位能力 | checkpoint 可用 |
| E02 | Stage 3 外观/quality 校准 | calibration | Brier/ECE/PR-AUC | 校准优于旧映射 |
| E03 | 校准后串行 | validation | IoU、误报、valid | 优于 E01/E00 |
| E04 | SingleScore 权重 | calibration 后 validation | IoU、排序 AUC | 误报不恶化 |
| E05 | R1 方向估计 | validation | R2 覆盖、中心误差 | IoU 绝对提升建议 >=0.01 |
| E06 | R2 自适应 FOV | validation | IoU、覆盖率 | 覆盖 >=99% |
| E07 | Fusor 框策略 | validation | IoU、宽高误差 | 分层无严重退化 |
| E08 | Geometry 修正 | validation | direct/indirect IoU | seam/极点测试通过 |
| E09 | PyTorch FP16 | validation | IoU 差异、P95 | 精度门槛通过 |
| E10 | GPU crop/搬运 | validation | 像素回归、P95 | 输出一致且更快 |
| E11 | batch 8 | validation | forward、显存 | forward 节省 >=25% |
| E12 | speculative 状态机 | validation | 接受/回退/覆盖 | 全部事务门槛通过 |
| E13 | speculative batch merge | validation | IoU、P95/P99 | 优于串行 Pareto 前沿 |
| E14 | TensorRT FP16 | validation | IoU、P95/P99 | 输出和精度门槛通过 |
| E15 | 冻结候选 | final holdout | 完整最终报告 | 只运行一次，不调参 |

每个实验失败后保留产物并记录失败原因，不覆盖同 ID。改变实现或参数时使用新 ID。

## 12. 代码实施位置与测试要求

### 12.1 校准和评分

- `src/instatarget/controller/fused_score.py`：版本化外观/运动映射和组合权重。
- `src/instatarget/tracker/pytorch_hit_session.py`：保留 raw logits、presence、quality、predicted IoU。
- `src/instatarget/tracker/observation.py`：字段完整传递和有限值验证。
- `src/instatarget/app/driver.py`：仅组合已冻结映射，不在热路径隐式拟合。

测试必须覆盖单调性、边界值、NaN/Inf、旧 checkpoint 兼容和校准产物 hash 不匹配。

### 12.2 Geometry 和 Fusor

- `src/instatarget/geometry/**`：seam、投影边界和 BFoV。
- `src/instatarget/controller/fusor.py`：候选聚合和框形状。
- `src/instatarget/controller/state_evaluator.py`：同一事务内两轮统一候选池。

测试必须覆盖 seam 两侧、极点、局部中心/边缘、超大 FOV、框顺序置换和重复 viewId 拒绝。

### 12.3 推测流水线

- `src/instatarget/app/speculative_scheduler.py`：严格 `4+4` 合并、TaskKey 路由和输出分区。
- `src/instatarget/controller/speculative_pipeline.py`：generation/revision、中心/尺度/覆盖校验和回退统计。
- `src/instatarget/tracker/backend.py`：混合 batch 与正式视图缓存隔离。
- `src/instatarget/app/driver.py`：后续接入正式事务和串行 fallback；在 validation 之前不得默认启用。

必须覆盖：默认关闭逐帧一致、R2/R1 乱序、缺输出、重复输出、非有限输出、generation/revision mismatch、LOST、sequence close、OOM、异常后无半事务提交。

### 12.4 评估工具

优先扩展 `tools/eval_manifest_controller.py`，不要另写多个口径不一致的临时脚本。建议输出：

```text
experiment.json
frames.jsonl
candidates.jsonl
timings.jsonl
calibration.json
summary.json
environment.json
```

汇总器必须验证 frameIndex 连续、数量相等、所有概率有限、split 与命令一致，再计算指标。

## 13. 失败处理和回滚

| 失败 | 处理 |
|---|---|
| Stage 3 输出字段缺失或非有限 | 不进入 calibration；保留 checkpoint 和错误报告 |
| calibration 提高 valid 但误报恶化 | 收紧 presence 映射/门限，不读取 holdout |
| IoU 提升仅来自更大框 | 检查宽高误差和 envelope inflation，拒绝该候选 |
| 小目标改善但 seam/极点退化 | 分层定位原因；未找到安全条件前不全局启用 |
| FP16 出现 NaN/Inf | 记录并回退 FP32；定位具体算子后单独修复 |
| batch 8 OOM | 使用逻辑流水线加两个 batch 4，不删视图、不降输入尺寸 |
| 推测覆盖不足 | 丢弃推测并从正式状态重跑 R1，不放宽阈值凑接受率 |
| TensorRT 输出漂移 | 停止发布 engine，保留 PyTorch 后端 |
| holdout 未达标 | 只报告失败，不用 holdout 继续调参；下一实验周期使用新的冻结流程 |

## 14. 最终冻结清单

进入 final holdout 前必须全部回答“是”：

- [ ] Stage 3 checkpoint、manifest、代码和依赖 hash 已记录。
- [ ] calibration 只使用 calibration split，参数已冻结。
- [ ] checkpoint 和 Controller 选择只使用 validation。
- [ ] circular ERP、spherical IoU、中心、尺度、误报和状态指标均已报告。
- [ ] seam、极点、边缘、小目标、快速运动、遮挡和 off-view negative 已分层检查。
- [ ] 性能报告包含阶段 P50/P95/P99、forward 数、显存、GPU 温度和 OOM。
- [ ] 串行 `4+4` fallback 保留且回归通过。
- [ ] speculative pipeline 默认关闭，只有通过门槛的配置才可显式打开。
- [ ] 每个正式 frame 最多提交一次，sink frameIndex 连续。
- [ ] final holdout 从未用于拟合、选择或调参。

## 15. 公开参考

以下资料用于实现方式和实验设计参考，不构成本项目性能证明：

- [HiT 官方实现](https://github.com/visionml/HiT)：模型结构、checkpoint 和单目标跟踪实现参考。
- [OSTrack 官方实现](https://github.com/botaoye/OSTrack)：one-stream 跟踪、评估和工程组织参考。
- [STARK 官方实现](https://github.com/researchmm/Stark)：时空跟踪与置信度/更新策略参考。
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)：`inference_mode`、内存格式和运行时性能建议。
- [NVIDIA TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)：FP16/INT8、动态 shape、engine 构建和性能测量要求。
- [Torchvision GIoU loss](https://pytorch.org/vision/stable/generated/torchvision.ops.generalized_box_iou_loss.html)：训练框回归损失语义参考。

外部仓库中的 FPS 通常使用普通透视图、单视图、不同 GPU、不同预处理和不同计时边界，不能直接与本项目 ERP crop、两轮多视图、球面投影和 Controller 的端到端 P95 比较。最终结论必须来自本项目固定数据、固定硬件和固定口径的 A/B。

## 16. 推荐的首批实际执行项

Stage 3 完成后，按以下最小闭环开始：

1. 冻结 checkpoint 和全部 hash；不读取 holdout。
2. 扩展 `eval_manifest_controller.py` 的多 sequence 汇总、AUC、分层和逐阶段计时。
3. E00/E01 已确认 Stage 3 原始定位与未校准分数分布。
4. 在 calibration split 完成 E02，冻结第一版 appearance/quality mapping。
5. 运行 E03/E04，先解决 valid rate、误报和候选排序。
6. 根据误差分解只选择 A2 到 A6 中贡献最大的一个方向做单变量 A/B。
7. 冻结 IoU 最优串行候选后再进行 E09 到 E14。
8. 只有全部冻结清单通过后才运行 E15。

阶段 0 至 4 已优先解决 Stage 3 分数重标定与 Fusor 框形状问题。FP16、数据搬运、batch 合并、推测流水线和 TensorRT 仍留在阶段 5，当前不得启用。

## 17. `PreEnhancement.md` 当前合规矩阵

本节区分“训练期间允许提前完成的代码机制”和“必须依赖 Stage 3、calibration、真实 GPU 或 final holdout 的验收”。不能因为机制代码已经存在，就宣称整份 `PreEnhancement.md` 已完成。

### 17.1 本分支已经完成的训练前准备

- 使用独立 `enhancement/preTraining` 分支，未修改 `src/instatarget/training/**`、`configs/train_stage3.yaml`、manifest、checkpoint 或训练依赖。
- `speculativePipeline.enabled=false`、`batchMergeEnabled=false`，现有串行 `4+4` 路径保持默认。
- 新增严格 `SpeculativePipelineConfig`；未知、缺失、非有限和不一致字段由 schema 拒绝。
- 新增不可变 `TaskKey`，包含 sequence、frame、attempt、view、generation 和 role。
- 新增严格 `R2(t)+speculative R1(t+1)` 四加四任务合并、输出身份校验、顺序恢复和角色分区。
- 混合 batch 只把正式 frame 视图写入 backend previous-view cache，推测视图不能污染模板来源。
- committed Controller 状态与 `SpeculativeState` 分离；推测对象没有 motion、StateMachine、公开结果、模板或 sink 的写入口。
- 已实现 generation/revision、stale、frame age、sequence、显式 LOST、方向置信度、中心差、log 尺度差、覆盖和有限值校验。
- 已实现空输出、路由不一致、非有限输出等唯一回退原因，以及接受率、回退率和按原因汇总。
- 已保留旧 `infer()` 接口和串行后端路径；默认关闭时不改变生产 Controller 阈值和 Geometry。
- 单元测试覆盖默认关闭、接受、generation/revision、LOST、中心/尺度/覆盖、sequence close、空/非有限输出、严格四加四、乱序恢复和 backend TaskKey 绑定。

### 17.2 尚未完成且不得伪报完成的项目

- 推测流水线尚未接入生产 `runTracking()` 的跨帧 lookahead、正式事务接受/重跑和异步取消；当前是默认关闭的机制预适配，不是可发布流水线。
- 尚未产生每帧完整 timing/diagnostic artifact；当前只有推测决策和汇总数据结构。
- 尚未完成 fake Driver 的跨帧接受、重跑、迟到输出、OOM 和半事务异常集成测试。
- 尚未在真实 RTX 4060 Laptop GPU 上测量 batch 4/8/10、CUDA Event forward、峰值显存和温度。
- Stage 3 checkpoint、独立 calibration、validation A/B 和阶段 4 A4 已完成；对应产物与聚合报告保存在 `artifacts/training_eval/`。final holdout 仍未读取。
- `candidateMinScore`、`fusionSourceMinConfidence`、外观/运动权重和推测门限均未重新拟合；这是正确的冻结状态。
- 推测接受率、回退率、覆盖率、IoU 变化和 P50/P95/P99 尚无真实结果。
- TensorRT FP16、GPU crop、INT8 和最终 30 Hz 服务目标尚未验证。

因此，本分支可以确认完成的是 Stage 3 期间的安全预适配和边界测试；不能确认整份 `PreEnhancement.md` 的最终交付清单已经全部完成。最终完成状态只能在第 4 节规定的后续阶段依次通过后给出。
