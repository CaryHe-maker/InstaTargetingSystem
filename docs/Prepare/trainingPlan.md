# HiT-Small 训练与工程改造蓝本

> 状态：Stage 1 至 Stage 3 已完成，Stage 3 最优 checkpoint 已集成生产串行路径。本文前半保留训练设计依据；当前运行参数以 `configs/RGBonly.yaml` 与版本化校准产物为准。

当前交付使用 `models/hit_small_stage3.pth`，训练结束于 step 6000。推理保留 learned presence、quality/predictedIoU 和 bbox；校准数据只来自 `E:\NewDownload\train\manifest.jsonl` 的 calibration split，最终 holdout 尚未读取。旧 checkpoint 加载、旧 score adapter 与旧 Beta 不再是兼容目标。

## 1. 目标与边界

目标是在现有 ERP/球面单目标跟踪系统上，提高目标存在判断、框定位和长期跟踪成功率，同时降低每帧端到端延迟。目标硬件为 RTX 4060 Laptop GPU，实测显存约 8 GiB；训练数据根目录为：

```text
E:\NewDownload\train
```

该目录下的视频全部视为本项目训练数据来源。它们不能同时作为最终测试集使用。如果其中包含官方比赛测试视频，必须移出训练流程并保持完全隔离。

本文只针对当前 360° RGB-only 跟踪路线：“RGB ERP -> 局部透视视图 -> HiT-Small -> 球面/ERP 回投”。训练、推理、评估和比赛提交均只使用 RGB，不包含其他输入模态或相关分支；系统输出球面方向、角尺寸和 BFoV。

## 2. 当前实现的事实与限制

1. Stage 1-3 训练循环、manifest dataset、增强、loss、分层 optimizer、checkpoint/resume 与验证均已实现。
2. Stage 3 使用 template `128x128`、search `256x256`，输出 corner bbox、presence logit 和 quality/predicted-IoU logit。
3. 生产 Tracker 固定使用第 0 帧 anchor；每个正常帧执行 TRACKING/UNCERTAIN 的串行 4+4，第二轮依赖第一轮结果。
4. `TemplateCache` 缓存预处理模板张量，不等于完整 backbone 特征缓存。
5. PyTorch FP32 是唯一冻结生产路径；ONNX/TensorRT 与 FP16 属于尚未开始的阶段 5。
6. 严格 `TrainingConfig` 与 `AppConfig` 分离；严格 Stage 3 校准产物另外绑定 checkpoint 和 Controller 工作点。
7. 所有真实训练、校准、validation IoU 与丢失率都以 `E:\NewDownload\train\manifest.jsonl` 为基础，仓库 `data/` 只用于纯单元测试。

## 3. 最终模型结构

保留 HiT 的空间化框预测能力，不把 bbox 改为全局池化 MLP。模型逻辑结构如下：

```text
template 128x128 + search 256x256
        -> HiT backbone
        -> FB neck / bottleneck
        -> spatial corner head -----------------> bbox (cx, cy, w, h)
        -> presence MLP ------------------------> p_present
        -> quality MLP -------------------------> p_quality / predicted IoU
```

### 3.1 Corner head

- 保留现有 HiT `Corner_Predictor` 的两张 corner heatmap 和 soft-argmax 机制。
- bbox 仍使用归一化 `cx, cy, w, h`，推理时再转换到局部像素。
- bbox 训练使用正样本 mask；目标不存在的样本不计算 bbox loss。
- 后续如增加 heatmap target，应使用可微 Gaussian/soft target，不能用不可微 argmax 替代训练路径。

### 3.2 Presence head

- 输入为 HiT 的对象/query embedding 或等价的 `output_embed`，不能直接对 ERP 全图做分类。
- 结构建议为 `LayerNorm -> Linear(256,128) -> GELU/SiLU -> Dropout(0.1) -> Linear(128,1)`。
- 输出必须是 raw logit，训练使用 BCEWithLogits 或 focal loss，推理再 sigmoid。
- 标签表示“目标是否在该局部 ViewSpec 中可见/可定位”，不是“Controller 最终是否接受该候选”。

### 3.3 Quality head

- 输入至少包含对象 embedding；可以拼接预测 bbox 的四个归一化量和 corner head 的稳定统计量。
- 输出一个 raw quality logit，目标为预测框与真值框的 IoU 或 `IoU >= threshold` 的质量概率。
- quality head 不得复用 Controller 的最终融合分数作为训练标签，避免状态机逻辑泄漏进模型。
- 对负样本，quality 目标为 0，但应降低其损失权重，避免质量头退化成第二个 presence head。

### 3.4 统一置信度语义

建议定义：

```text
p_model = calibrate(p_present * p_quality)
```

`p_model` 才作为 Tracker 的外观置信度输入。模型原始 logits、`p_present`、`p_quality`、预测 IoU 和校准后的 `p_model` 必须同时记录，便于定位误差。

旧的 heatmap entropy 分数可以保留为诊断字段，但不能继续作为主要置信度。新权重发布后，外观校准、运动校准、`candidateMinScore` 和 `fusionSourceMinConfidence` 必须在独立 calibration split 上重新拟合。

## 4. 冻结和解冻策略

### 4.1 推荐最终策略

冻结：

- HiT patch embedding/stem。
- HiT 前两个层级的 backbone blocks。
- 早期 BatchNorm 的 running statistics。

训练：

- HiT 最后一个层级的 blocks。
- `bottleneck`/FB neck。
- 现有 corner head。
- 新 presence head。
- 新 quality head。

当前 checkpoint 的键名中最后层级候选通常对应 `backbone.body.blocks.18:` 之后，但实现时必须根据 `AttentionSubsample` 的真实边界自动确定，不能永久硬编码 18。必须打印并保存每个参数组的名称、可训练参数量和学习率。

### 4.2 明确禁止的策略

- 不要冻结深层、只训练浅层。浅层是通用纹理/边缘特征，改变浅层而固定深层会造成特征分布错位。
- 不要一开始全量解冻 11M HiT。目标视频数量和场景多样性未验证前，过拟合和灾难性遗忘风险过高。
- 不要只训练两个 MLP 就把结果当作最终商业模型；它只能作为管线和标签正确性的基线。

### 4.3 分阶段训练

**Stage 0：数据和基线检查**

- 不更新权重。
- 跑现有权重，保存局部框、球面 IoU、中心误差、状态分布、每帧视图数、P95 延迟和显存。
- 确认训练 manifest、局部 crop 和标签可视化正确后才能进入下一阶段。

**Stage 1：新头 smoke test**

- 冻结全部 HiT、neck 和 corner head，只训练 presence/quality head。
- 目标是确认正负标签、loss 下降、输出概率单调、负样本不会全部输出高分。
- 训练步数只需达到验证集稳定，不以此阶段的最终 IoU 做结论。

**Stage 2：冻结 backbone 的头部适配**

- 解冻 FB neck、corner head、presence head、quality head。
- backbone 全部冻结，BN 保持 eval。
- 这是第一个可用于端到端 Controller 验证的候选权重。

**Stage 3：推荐发布候选**

- 在 Stage 2 最优 checkpoint 上解冻最后一个 HiT 层级。
- 采用分层学习率：新头最高，neck/corner 次之，最后层级最低。
- 使用 sequence-level validation early stopping，防止只对某个视频过拟合。

**Stage 4：可选实验**

- 只有 Stage 3 在独立验证集仍明显欠拟合时，才解冻中间层级。
- 必须单独保存、评估和可回滚，不能直接覆盖 Stage 3 权重。

## 5. `E:\NewDownload\train` 数据处理规范

### 5.1 先确认视频是否有标签

视频本身不能提供监督框。如果目录中没有与帧对应的 instance mask、bbox 或人工标注，必须先建立标注流程：

- 优先人工标注关键帧 bbox/mask，再用可靠的视频传播工具生成中间帧候选。
- 所有遮挡、目标消失、目标重新出现、快速运动、经线跨越和相似干扰物必须人工复核。
- 伪标签必须记录 `labelSource`、质量等级和是否经过人工确认；不能把伪标签无标记地当真值。

### 5.2 Manifest 最小字段

每个 frame/target 记录至少包含：

```text
sequenceId, videoPath, frameIndex, timestamp
targetInstanceId, bbox or mask, visible, occluded, truncated
width, height, labelSource, labelQuality, split
```

训练代码应先读取 manifest，再按需解码视频。不要依赖当前 `__getitem__` 从视频开头重复扫描。

### 5.3 数据划分

- 按 sequence/场景/目标划分，不按随机帧划分。
- 推荐 train 70%、validation 15%、calibration 5%、final holdout 10%。
- 同一视频的相邻帧不能出现在不同 split。
- `final holdout` 只在所有模型和阈值冻结后使用。
- 如果视频数量不足以完成 sequence-level split，必须增加视频或降低模型复杂度，不能用随机帧伪造独立验证集。

### 5.4 Template/search pair

- template 使用稳定、无遮挡、框完整的目标帧；同时模拟生产中的固定首帧 anchor。
- search frame gap 必须覆盖相邻帧、短时运动和长时间间隔。
- 训练局部视图必须调用生产 Geometry，覆盖 30°–120° 动态 FOV、120° UNCERTAIN FOV、目标边缘、极点、经线和严重透视拉伸。
- 正样本应包含中心、边缘、部分可见和大尺度变化。
- 负样本包括目标不在视图、目标消失、错误 instance、同类干扰物和强遮挡。
- 每个 batch 不能只有容易的中心正样本；必须记录正负比例和各类困难样本比例。

### 5.5 增强

建议使用与视频域一致的亮度/对比度、色温、压缩、模糊、运动模糊、噪声、轻微缩放和遮挡增强。增强必须同步修改 bbox/mask。不要使用未经验证的强增强破坏全景经线或局部透视语义。

## 6. Loss、采样和训练超参数

基础 loss：

```text
L = λp * PresenceLoss
  + positive * (5 * L1(box) + 2 * GIoU(box))
  + λq * QualityLoss
```

建议初始值 `λp=1`、`λq=1`；通过 validation 的 Brier、ECE、PR-AUC 和定位指标联合调整。类别极不平衡时优先使用 focal loss 或受控正负采样，不要单纯把 presence threshold 调高。

训练配置至少包括：manifest、split、pair 采样规则、FOV 分布、负样本比例、增强、batch size、gradient accumulation、学习率、每组参数学习率、weight decay、warmup、scheduler、epoch/step、AMP、随机种子、worker、checkpoint、resume、验证周期和 early stopping。

4060 8GB 起始建议：AMP/BF16，物理 batch 8，gradient accumulation 4，有效 batch 32；OOM 时使用 batch 4、accumulation 8。优化器使用 AdamW，`weight_decay=1e-4`，gradient clip `0.1`。所有数值必须通过 1000-step profiler 校准，不能假设官方 32 batch 能在本机运行。

## 7. 必须修改的代码边界

后续实现按以下顺序修改，不要把训练逻辑塞入运行时文件：

1. `training/dataset.py`：加入 manifest、索引缓存、sequence split、template/search pair、生产 Geometry crop、正负样本和可复现采样。
2. 新增训练数据/增强模块：视频按需解码，局部 crop 可选择分片缓存，保存 label quality 和困难样本类型。
3. `training/losses.py`：实现 presence、bbox L1/GIoU、quality loss，并返回每个子损失和正负样本统计。
4. `training/train_backend.py`：实现模型构建、冻结策略、分层 optimizer、AMP、梯度累积、验证、checkpoint、resume、early stopping 和日志。
5. `configs/train_backend.yaml`：改为完整 TrainingConfig；不能直接复用 `configs/RGBonly.yaml`。
6. `core/config.py`：增加严格的 TrainingConfig schema、范围检查和路径解析测试。
7. HiT model wrapper：增加显式 `presenceLogit`、`qualityLogit`、bbox/heatmap 输出，移除对 forward hook 作为正式接口的依赖。
8. `tracker/hit_backend.py`、`pytorch_hit_session.py`、`tracker/observation.py`：扩展 prediction/observation 字段，保持 raw logits、概率、质量和 bbox 语义清晰。
9. `controller/score_calibration.py` 与 `fused_score.py`：加载 checkpoint 绑定校准产物；保留 Stage 3 原始 presence/quality/乘积分数作诊断，不引入旧模型适配。
10. `app/driver.py` 和 `StateEvaluator`：先使用新 `p_model` 排序，再通过独立验证调整运动融合和接受门限；不能把 Controller 是否接受作为训练标签。
11. `eval/`：增加 presence PR-AUC、ROC-AUC、Brier、ECE、quality calibration、目标消失误报、重捕获帧数和 P95/P99 延迟。
12. `tests/`：覆盖 loss mask、负样本、输出顺序、Stage 3 checkpoint 严格加载、冻结参数集合、概率范围、跨缝/极点局部 crop、事务两轮和校准产物拒绝性测试。
13. vendor/packaging：解决 `src/instatarget/vendor/hit/lib` 的 git 忽略、wheel/Docker 包含、许可证和权重发布问题。

## 8. 推理和速度优化顺序

精度候选权重冻结前，不进行激进的量化或换主干。推荐顺序：

1. 用新 presence/quality 进行条件早停：高质量 TRACKING 候选可以跳过第二轮；不确定时保留完整 4+4 fallback。
2. 统计 crop、RGB preprocessing、HiT forward、projection、calibration、Controller 各阶段的 P50/P95 时间和 GPU 同步方式。
3. 缓存固定输出尺寸/FOV 的射线网格；必要时将透视采样迁移到 GPU。
4. 实现 ONNX/TensorRT FP16，并对 batch 1/4/10 分别验证数值和速度。
5. 最后才尝试 INT8；必须使用代表性校准集并通过 spherical IoU、负样本 FPR 和非有限值回归。
6. 长序列外观漂移验证通过后，再启用受门控的 recent template；anchor 必须始终保留并支持回滚。

任何早停、模板更新或量化都必须保留完整视图搜索恢复路径，不能为了平均 FPS 删除找回能力。

## 9. 验收指标与发布门槛

候选 checkpoint 必须先在固定 validation 上比较；只有全部内容冻结后才允许在 final holdout 上报告一次：

- circular ERP IoU、BFoV spherical IoU、success AUC、success@0.5。
- 中心大圆角误差、框宽高相对误差、`envelopeInflation`。
- presence PR-AUC/ROC-AUC、Brier、ECE、quality calibration。
- 目标缺失时误报率、tracking loss rate 与 lost frame count。LOST/找回尚未实现，当前不能报告重捕获算法收益。
- valid rate、每状态帧数、平均视图数、每帧 forward 数。
- end-to-end P50/P95/P99、GPU 利用率、峰值显存和温度稳定性。

当前硬件的初始服务目标是端到端 P95 不超过 33ms，即稳定 30Hz；最终精度目标必须先由现有权重建立基线，再要求 Stage 3 在不超过 10% 延迟增加的情况下提升定位/存在指标并降低缺失帧误报。未同时满足精度、误报和 P95 的权重不得发布。

## 10. 运行前和提交前提醒

- 训练视频目录不是自动等于标注数据；没有标签时必须先标注并审核。
- 正式测试视频禁止参与训练、阈值选择或 calibration。
- 训练/验证不能按相邻帧随机拆分。
- 不要把 `stateScore`、Fusor 结果、最终 valid 标志作为 presence 标签。
- 不要把运动预测写入训练真值或运动历史。
- 生产只接受 Stage 3 `model` state；旧 checkpoint 与旧参数适配已删除，A/B 依赖独立保存的历史实验产物而不是生产兼容分支。
- 新字段加入 Core 类型后必须同步 producer、consumer、配置 schema、文档和测试。
- 发布 checkpoint 时分离训练恢复文件和推理文件；推理文件只保留模型权重和必要元数据。
- Docker/competition 镜像必须在无源码挂载、无联网条件下加载新权重。
- 每次实验保存代码版本、manifest hash、模型初始权重 hash、随机种子、配置、校准参数和完整指标。

## 11. 推荐执行顺序

```text
现有权重基线
 -> 视频/标注审计
 -> manifest + 生产 crop 可视化
 -> Stage 1 head-only
 -> Stage 2 neck + corner + heads
 -> Stage 3 最后 HiT 层级 + neck + heads
 -> 独立 calibration
 -> Controller A/B
 -> 条件早停
 -> TensorRT FP16
 -> 可选 recent template / INT8 / LiteTrack 替换实验
```

在 Stage 3 和独立 calibration 完成前，不应修改状态机阈值、Fusor 几何或同时引入新的主干网络；否则无法判断收益来自训练还是来自运行时规则变化。
