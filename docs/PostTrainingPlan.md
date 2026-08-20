# Stage 3 后续优化计划

## 1. 当前冻结基线

本文记录 Stage 3、独立 calibration 和阶段 4 A4 完成后仍未实施的工作。当前生产基线必须保留为所有后续实验的对照组：

- 模型：`models/hit_small_stage3_inference.pth`，与原始 Stage 3 `model` state 逐张量一致，FP32。
- 外观输入：`presence * predictedIoU`，使用 checkpoint 绑定的单调 Beta calibration。
- SingleScore：appearance/motion `0.50/0.50`。
- Controller 工作点：`candidateMinScore=0.597262`、`fusionSourceMinConfidence=0.740642`。
- 融合几何：`best_source`。
- 普通帧：顺序两轮 `4+4`，每帧 2 次 forward；speculative pipeline 关闭。
- validation 四序列、1796 个可见帧：circular ERP mean IoU `0.35915`、success@0.5 `0.30178`、spherical mean IoU `0.29694`、tracking loss rate `0.11971`（215 帧）、P95 `355.80 ms`、P99 `384.95 ms`。
- `LOST` 自动转移和失踪找回尚未启用；丢失率目前只用于离线统计。
- final holdout 尚未读取。

所有真实 calibration、IoU、丢失率和性能 A/B 只能使用 `E:\NewDownload\train\manifest.jsonl` 中相应的 calibration/validation split。仓库 `data/` 只允许纯单元测试。每次实验只改变一个主要变量，保留逐帧、逐候选、timing、环境和 hash 产物；未冻结前不得读取 holdout。

## 2. 大概率正收益、优先执行

本节项目不改变公开跟踪语义，或具有明确回退路径。仍需 A/B，不能因为风险较低就直接发布。

### 2.1 补齐分阶段 profiler

扩展 `eval/profiler.py` 和 manifest 评估产物，逐帧记录 decode、crop、preprocess、host-to-device、CUDA Event forward、projection、calibration、Controller 和 total 的 P50/P95/P99，同时记录 batch size、forward count、显存、温度和 OOM。

这是所有速度工作的前置条件。计时开关本身需做 overhead A/B；关闭 profiler 时不得改变结果。

### 2.2 低风险 PyTorch 推理优化

按单变量顺序测试：

1. 固定 shape 的 cuDNN benchmark。
2. channels-last 模型与输入。
3. 复用固定 shape 的 CPU/GPU buffer。
4. pinned memory 与 non-blocking copy 配对。
5. 减少重复 NumPy/Tensor 分配和不必要同步。

每项必须逐输出比较 bbox、presence、quality、SingleScore 和最终 TrackResult。通过条件为 circular ERP IoU/success@0.5 不下降、无新增非有限输出且 P95 或明确的阶段耗时下降。

### 2.3 PyTorch FP16，保留逐批 FP32 回退

固定 Stage 3 checkpoint、视图数、输入尺寸、calibration 和 Controller 参数，比较 FP32/FP16。现有非有限检查必须覆盖 bbox、corner heatmap、presence/quality logits 和概率；任一异常整批回退 FP32并计数。

FP16 只有在 validation 精度满足门槛且端到端 P95 明显下降时才可进入候选。建议门槛：mean IoU 相对下降不超过 `0.5%`、success@0.5 不下降超过 `0.5` 个百分点、loss rate 和 absent FPR 不恶化。

### 2.4 完善 validation 覆盖与丢失事件报告

当前四序列没有 absent frame，不能据此证明 absent FPR 为零。应扩大 validation 序列覆盖，按 sim/real、目标大小、速度、遮挡、seam、纬度、FOV 和 view-edge 分层，并把零 IoU 连续帧合并为 loss episode，新增：

- loss episode 数、长度 P50/P95/最大值；
- 丢失前 presence/quality/motion/SingleScore 轨迹；
- 首次零 IoU 帧的 R1/R2 覆盖、中心误差和尺度误差；
- 在不改变状态机的前提下，shadow `lostCandidate` 触发准确率。

这一步仍只统计，不实现 LOST/recovery，可为后续找回算法提供不会泄漏 holdout 的触发依据。

### 2.5 校准与产物发布自动校验

把 checkpoint、calibration、YAML、Docker 压缩 checkpoint 的哈希核对加入提交前脚本/CI；检查 GitHub 克隆后的 Docker build context 必含压缩权重与配对校准，且最终镜像恰好 7 个文件系统层。该项主要降低发布失败概率，不改变模型输出。

### 2.6 有证据时的小范围 Geometry 修正

先计算 local IoU、direct circular ERP IoU、spherical IoU 与 `envelopeInflation` 的相关性。仅在误差明确来自包络时，修正最短循环水平区间或边界采样实现；必须保持 BFoV 球面语义，并增加 seam、极点和大 FOV 回归。没有证据时不修改 Geometry。

## 3. 有风险的优化，必须隔离实验

这些工作可能提高精度或速度，但会改变搜索覆盖、状态、模板、输出几何或执行顺序。必须从低风险项目取得可靠 profiler/分层基线后再做。

### 3.1 LOST 状态与失踪找回

当前 `11.97%` tracking loss rate 说明潜在收益较大，但错误触发会破坏连续跟踪。建议分三步：

1. shadow lost detector：只记录连续低 SingleScore、零/近零 IoU 的可观测代理、R1/R2 无覆盖和运动不确定度，不改变状态。
2. 离线回放选择触发/退出条件，按 loss episode 衡量检测延迟、误触发率和可恢复比例。
3. 才接入正式 `LOST` 与 cubemap/局部恢复计划；恢复候选需连续确认后提交，anchor 模板保持不变。

验收必须同时报告 loss rate、恢复成功率、恢复帧数、正常帧误触发率、absent FPR、视图数和 P95/P99。禁止用真值 IoU 作为线上 LOST 输入。

### 3.2 A2：Round 1 多候选方向融合

比较当前最佳候选中心、top-k calibrated SingleScore 球面加权中心，以及只对重叠可靠来源加权。可能提高 R2 覆盖，也可能被多个错误候选拉偏。Round 1 不得更新正式运动、尺度、模板或公开结果。

### 3.3 A3：R2 自适应 FOV

可靠目标缩小 FOV 可提高局部像素分辨率，运动不确定时扩大 FOV可提高覆盖。但快速运动、小目标和校准误差可能使目标直接离开视图。只有 R2 覆盖率 `>=99%`，且各目标大小/速度/纬度分层无明显退化时才能启用。

### 3.4 A5/A6：投影与尺度时间滤波

Geometry 修正若缺少误差证据会改变所有框；尺度滤波可能降低抖动，也会在目标快速接近/远离时产生滞后。尺度滤波只能使用正式接受的 Round 2 测量，在 log 宽高空间按 calibrated quality 与不确定度加权，且必须比较宽高误差 P50/P95 和快速尺度变化分层。

### 3.5 A7：recent/stable template 更新

只允许正式 Round 2 高质量结果更新 recent template；UNCERTAIN、LOST、遮挡、低 presence 或尺度跳变时禁止更新。frame 0 anchor 永不覆盖并可立即回滚。该项有明显漂移和错误自强化风险，优先级低于 LOST 诊断、覆盖和尺度问题。

### 3.6 GPU crop/resize

将 ERP 透视 crop、resize 和预处理搬到 GPU 可能降低 CPU 与传输开销，但插值、坐标原点、边界填充、颜色通道和 seam 处理的细小差异会改变模型结果。必须先像素级回归，再做 bbox/IoU A/B；不得与 FP16 同次改变。

### 3.7 batch 8 与 speculative pipeline

先分别 profile 两个 batch 4、一个 batch 8、一个 batch 10 和逻辑合并但物理两个 batch 4。batch 8 forward 节省不足 `25%` 时不接入复杂流水线。

推测流水线必须先用 fake backend 验证 generation/revision、乱序、迟到丢弃、异常、OOM、sequence close 和一次正式提交；再在 validation 上仅启用状态机，最后才启用 batch merge。目标为接受率 `>=80%`、回退率 `<=20%`、覆盖率 `>=99%`，同时 IoU、success@0.5、loss rate、absent FPR 和 P95/P99 不恶化。

### 3.8 条件跳过第二轮

高质量 TRACKING 帧跳过 Round 2 有较大速度潜力，但会直接改变当前 `4+4` 精度语义，并可能放大 quality 过度自信。只能在 calibration 可靠、R1 框与完整 R2 结果高度一致的分层上试验；必须保留立即执行完整第二轮的 fallback。这项应晚于 FP16、搬运优化和 batch profile。

### 3.9 TensorRT FP16/INT8

TensorRT FP16 仅在 PyTorch FP16 和固定 shape profile 稳定后开始，逐输出比较 corner、presence、quality 和 bbox，并复用同一 calibration/Controller。INT8 最后考虑，需要代表性 calibration 数据并重新验证 seam、高纬度、负样本和非有限输出。engine 必须绑定 TensorRT/CUDA/驱动/GPU/builder flags，不提交不可复现的本机 engine。

### 3.10 更大输入、mask/refinement、旋转框或重新训练

384/512 输入、窄 FOV 二次 refinement、mask head、旋转框、主干替换或 Stage 4 再训练都可能改善特定误差，但会显著增加算力或重新引入训练/校准变量。只有分层误差证明现有 Stage 3 表达能力是主要瓶颈时才建立独立训练周期；不得与 Controller、Geometry 或后端优化混合归因。

## 4. 推荐执行顺序

```text
P0 profiler
 -> validation/absent/loss episode 覆盖
 -> cuDNN/channels-last/buffer/pinned-memory 单变量 A/B
 -> PyTorch FP16
 -> 依据 profiler 决定 GPU crop
 -> batch 4/8/10 profile
 -> shadow LOST detector
 -> LOST/recovery 离线回放与受控接入
 -> A2/A3/A5/A6 单变量精度实验
 -> speculative pipeline
 -> conditional Round 2
 -> TensorRT FP16
 -> recent template / INT8 / 更大模型等高风险实验
 -> 全部冻结后一次 final holdout
```

任何候选失败都保留实验产物和失败原因，恢复当前 Stage 3 FP32 串行 `4+4` 基线。final holdout 只做一次最终报告，不能用于选择本计划中的任何项目。
