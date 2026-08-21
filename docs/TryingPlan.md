# Stage 3 危险优化独立实验计划

## 编号完成度

| 编号 | 状态 |
| --- | --- |
| 1.0 | 完成，当前冻结基线 |
| 2.1 | 完成 5-seq 预筛，全部回退；4x 仅单序列诊断 |
| 2.2 | 未进行 |
| 2.3 | 未进行 |
| 2.4 | 未进行 |
| 2.5 | 未进行 |
| 2.6 | 未进行 |
| 2.7 | 未进行 |
| 2.8 | 未进行 |

`1.0` 表示当前生产基线已经冻结并作为所有实验的共享对照。`2.1` 至 `2.8` 是相互独立的实验集；每组都要单独统计、单独判定、单独保留产物，不把一个实验组当成下一个实验组的串行前置步骤。

## 本轮执行结论（2.1 ERP 已补充）

本节预留给实验完成后的汇总。测试前只登记实验范围、版本、数据和失败原因，不预先填写结论。

### 总体结论

- 共享生产基线：`E:\tringData\shared_control\production`，Stage 3 FP32 顺序 `4+4`。
- 通过门槛的实验集：当前无；2.1 ERP 没有变体通过硬回归门槛。
- 保留到候选实现的变体：当前无。
- 明确回退的变体：`erp_crop_2x_strict`、`erp_crop_2x_relaxed`。
- 未完成或不具备比较资格的变体：`erp_crop_4x_relaxed` 仅完成 `seq_0045` 且同时改变第一轮 FOV；`erp_crop_2x_3x_best` 未运行；2.2 至 2.8 未进行。
- 是否读取 holdout：否

### 实验结果登记

| 实验集 | 变体 | 完整性 | Mean IoU | Spherical IoU | Success@0.5 | Loss rate | Absent FPR | P95 | 结论 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2.1 ERP 局部裁图 | `erp_crop_2x_strict` | 5/5 预筛完整，非完整 15-seq | 0.277825 | 0.220264 | 25.24% | 31.45% | 0.00% | 353.24 ms pooled | 回退；`seq_0017` 严重回归 |
| 2.1 ERP 局部裁图 | `erp_crop_2x_relaxed` | 5/5 预筛完整，非完整 15-seq | 0.327600 | 0.270956 | 33.50% | 28.78% | 4.62% pooled | 334.04 ms pooled | 回退；loss 和硬回归门槛失败 |
| 2.1 ERP 局部裁图 | `erp_crop_4x_relaxed` | 仅 `seq_0045`，组合变量 | 0.206975 | 0.205029 | 20.05% | 35.12% | 33.53% | 334.98 ms | 仅诊断，不具备晋级资格 |
| 2.1 ERP 局部裁图 | `erp_crop_2x_3x_best` | 未运行 | — | — | — | — | — | — | 暂停；先解决 2x 路径硬回归 |
| 2.2 目标模板 | `template_strict` | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| 2.2 目标模板 | `template_relaxed` | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| 2.3 Fusor 几何 | `fusor_weighted_box` | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| 2.3 Fusor 几何 | `fusor_robust_spherical_consensus` | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| 2.4 自适应 FOV | `fov_adaptive_both_rounds` | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| 2.4 自适应 FOV | `fov_adaptive_round1_only` | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| 2.5 IoU refinement | `iou_refine_head` | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| 2.6 Identity verifier | `distractor_identity_verifier` | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| 2.7 局部/全局恢复 | `local_global_recovery_verifier` | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| 2.8 流水线与 GPU Geometry | `pipeline_only` | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| 2.8 流水线与 GPU Geometry | `gpu_geometry_only` | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| 2.8 流水线与 GPU Geometry | `pipeline_gpu_geometry` | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |

### 测试后需要补充的内容

- 每个变体相对共享对照的逐序列变化，特别是 `train_sim/seq_0017` 和 `train_sim/seq_0045`。
- 每个变体的完整性、缺帧、失败重试、OOM、非有限值和 hash 校验结果。
- 触发子集与全体帧的精度结果，避免用少量触发帧的收益代表整体收益。
- 各阶段 P50/P95/P99、视图数、forward 数、峰值显存、CPU/GPU 同步和数据类型转换次数。
- 失败原因、回退原因、保留/淘汰决定以及是否允许进入下一次独立验证。

## 1. 当前冻结基线

所有实验必须以同一个共享生产对照为参照。基线只描述当前实现，不因某个实验集的结果自动改变。

- checkpoint：`models/hit_small_stage3.pth`，FP32。
- calibration：checkpoint 绑定的 Stage 3 Beta calibration。
- 外观输入：`presence * predictedIoU`。
- SingleScore：appearance/motion `0.50/0.50`。
- Controller 工作点：`candidateMinScore=0.597262`、`fusionSourceMinConfidence=0.740642`。
- Fusor：overlap `0.70`，生产几何为 `best_source`。
- 模板：第 0 帧 anchor 永久保留，不被覆盖。
- 正常线程：只运行 `TRACKING` 和 `UNCERTAIN`。
- 普通帧：顺序两轮 `4+4`，每帧两次 HiT forward。
- 第二轮：中心依赖第一轮 Fusor 结果，不能在保持语义不变的前提下强行合并为一轮。
- LOST：不由正常状态机自动触发，仅保留为显式隔离实验能力。
- speculative pipeline：关闭。
- Geometry：当前 CPU 路径，保持现有 ERP seam、极点、FOV 和 BFoV 语义。

真实 calibration、IoU、loss rate、absent FPR 和性能统计只能使用 `E:\NewDownload\train\manifest.jsonl` 的对应 split。仓库 `data/` 只用于纯单元测试；所有实验禁止读取 holdout。

## 2. 独立实验总则

### 2.1 独立性

每个实验集都可以单独启动、单独完成、单独聚合和单独回退。不得要求先完成 2.1 才能运行 2.2，也不得把“上一个实验已经启用的开关”当成后一个实验的默认输入。

每个变体都必须创建新的 Runtime、Controller、TemplateCache 和 HiT session。实验之间不得共享运行时状态、模板 revision、运动历史、CUDA stream、队列中的帧或缓存 tensor。允许复用已经通过 hash 校验的共享生产对照产物，但不得重新解释或覆盖它。

推荐的单实验入口形态如下，具体脚本名可以调整：

```powershell
& ".venv\Scripts\python.exe" "tools/run_experiment.py" `
  --experiment "pipeline_geometry" `
  --variant "gpu_geometry_only" `
  --manifest "E:\NewDownload\train\manifest.jsonl" `
  --dataset-root "E:\NewDownload\train" `
  --output-root "E:\tringData" `
  --resume
```

该入口一次只运行一个实验变体。可以另写外层批处理工具，但不得把多个实验包装成必须按顺序连接的单一实验命令。

### 2.2 固定数据和序列

所有实验集使用同一份 validation 序列、帧范围、初始化和版本记录。序列清单固定为：

```text
train_sim/seq_0045
train_sim/seq_0017
train_real/seq_0018
train_real/seq_0026
train_real/seq_0037
train_real/seq_0005
train_real/seq_0036
train_sim/seq_0078
train_sim/seq_0010
train_sim/seq_0048
train_sim/seq_0076
train_sim/seq_0075
train_sim/seq_0036
train_sim/seq_0052
train_sim/seq_0058
```

`train_sim/seq_0017` 是局部跟踪链的硬回归序列，`train_sim/seq_0045` 是含 absent 区间的硬回归序列。不得根据中间结果替换序列或帧范围。

### 2.3 统一记录

每个实验变体至少记录：

- circular ERP mean IoU、AUC、success@0.5；
- spherical mean IoU、球面中心误差、BFoV 宽高误差；
- tracking loss rate、零 IoU 帧、loss episode 数及长度；
- absent FPR、valid rate；
- TRACKING/UNCERTAIN 帧数、状态转移、StateScore 和 SingleScore 分布；
- 每帧视图数、每轮 batch size、forward 数；
- decode、CPU crop、GPU crop、resize、preprocess、H2D、CUDA forward、D2H、projection、calibration、Controller 和 total；
- 峰值显存、温度、OOM、非有限输出、FP32 fallback；
- 逐帧 `TrackResult`、逐轮/逐视图候选、实际分支、实验专用字段；
- Git commit、dirty 状态、配置、checkpoint/calibration hash、环境和运行命令。

所有数值必须检查有限性。聚合必须同时提供 macro、按帧 micro 和逐序列结果。

### 2.4 独立实验的共同验收

每个变体先通过短测，再运行固定 validation：

1. fake backend 能建立和关闭完整 Runtime，且状态不会跨序列或跨变体泄漏。
2. `--max-frames 3` 能生成逐帧结果、候选、timing 和 `_SUCCESS.json`。
3. 中断后 `--resume` 只恢复未完成任务，hash 改变后拒绝复用旧成功标记。
4. 聚合器拒绝缺帧、重复帧、非有限值、缺少实验字段和对照 hash 不一致的输入。
5. 实验失败必须保留错误和部分产物；不能静默删除失败序列后宣布候选胜出。

## 3. 实验一：ERP 原图局部框

### 3.1 目的和对照

验证目标远离 ERP 边界时，第二轮直接裁剪 ERP 原图是否能减少透视投影误差。对照使用共享生产方案：两轮均为 Geometry 透视 crop，第二轮围绕第一轮 Fusor 最佳中心生成 Type1 四角视图。

`size0` 是上一帧最近一次被 Controller 接受的稳定 ERP bbox 尺寸。没有最佳候选、没有合法 `size0`、裁剪无效或尺寸过小时，必须回退生产路径。

### 3.2 变体

- `erp_crop_2x_strict`：宽高各放大 `2.0` 倍的 ERP 矩形完全在图像内时，第二轮只做普通矩形 crop + resize；否则回退。
- `erp_crop_2x_relaxed`：先用 `1.25` 倍框判断是否可执行；通过后实际使用 `2.0` 倍 crop，超界部分裁去；否则回退。
- `erp_crop_4x_relaxed`：先用 `2.0` 倍框判断是否可执行；通过后实际使用 `4.0` 倍 crop，超界部分裁去；否则回退。TRACKING/UNCERTAIN 第一轮 Type1 均固定为 `120° x 120°`。
- `erp_crop_2x_3x_best`：通过 `1.25` 倍判断后同时产生 `2.0` 倍和 `3.0` 倍 crop，组成一个 batch，按 calibrated SingleScore 选择较高者；同分固定选择 `2.0` 倍。

直接 ERP 分支不得与第一轮候选融合，不得绕过 candidate gate、valid gate、measurement acceptance 或状态机。ERP x 坐标必须使用循环 seam 语义，不能用普通平面 min/max。

### 3.3 专用记录

记录 `ERP_rate`、触发/回退率、crop 原始尺寸、resize 比例、目标在 crop 中的位置、直接 bbox 与生产回投 bbox 的差异，以及 crop/resize/forward/映射耗时。单独报告触发子集和全体帧。

### 3.4 已执行结果和决定

完整结果见 [ERPChangeReport.md](ERPChangeReport.md)。

- `erp_crop_2x_strict` 和 `erp_crop_2x_relaxed` 均完成 5-seq 预筛，10/10 任务成功且产物有限、hash 完整。
- strict/relaxed 在 `train_sim/seq_0017` 的 Mean IoU 分别从对照 `0.269560` 降至 `0.035852` 和 `0.101052`，loss rate 分别升至 `85.39%` 和 `69.73%`，违反硬回归门槛。
- relaxed 虽在 `seq_0045` 和 `seq_0078` 有收益，但不能抵消 `seq_0017` 与 `train_real/seq_0005` 的严重退化。
- `erp_crop_4x_relaxed` 仅在 `seq_0045` 完成组合诊断；结果优于生产对照但 absent FPR 明显差于 `2x_relaxed`，且未运行 `seq_0017`，不具备晋级资格。
- 实验 2.1 最终决定为全部回退，继续使用共享生产第二轮 Geometry Type1 路径；不读取 holdout。

## 4. 实验二：受控更换 HiT 目标模板

### 4.1 目的和固定语义

验证稳定条件下的 recent template 是否适应外观、尺度和照明变化，同时避免漂移。第 0 帧 anchor 的图像和特征永久保存。

候选模板只能来自最终已提交、`valid=true`、通过 measurement acceptance 的结果。候选在下一帧与图 0 做无副作用双分支验证；验证帧公开图 0 分支，候选通过后从再下一帧开始使用。

### 4.2 变体

- `template_strict`：连续 3 个已提交帧为 TRACKING，当前 StateScore `>0.8`，且在最近 10 帧中排名前 2；下一帧图 0/候选 circular ERP IoU `>=0.5` 才提升。
- `template_relaxed`：连续 2 个已提交帧为 TRACKING，当前 StateScore 在最近 10 帧排名前 2；验证 IoU 门槛为 `0.3`。

连续 3 帧 UNCERTAIN 时切回图 0。新模板始终与图 0 验证，不与动态模板链式验证。任何分支不得污染正式 Controller、运动历史或公开结果。

### 4.3 专用记录

记录 `PhotoChangeRate`、候选生成/验证/提升/拒绝/回滚次数、验证延迟、动态模板持续帧数、图 0/候选差异、使用非图 0 模板帧的独立指标和双分支额外成本。

## 5. 实验三：Fusor 最终框几何

### 5.1 目的和对照

只改变融合候选胜出后的 bbox/BFoV 几何，不改变 candidate 集合、SingleScore、overlap `0.70`、source confidence、视图计划、状态机或模板。若最终胜出的是单框，所有变体必须输出相同结果。

### 5.2 变体

- `fusor_weighted_box`：以 calibrated SingleScore 为权重，在 seam-aware 连续 ERP 坐标中加权中心；宽高在 log 空间加权；非有限或无效时回退 `best_source`。
- `fusor_robust_spherical_consensus`：以胜出 pair 为 seed，吸收与 seed overlap 足够的高分来源，在球面切平面执行 3 轮 Huber 重加权；角尺寸使用加权中位数并限制在 support 范围内；support 不足或退化时回退。

### 5.3 专用记录

记录融合胜出率、实际生效率、回退率、support 数、输出与来源框的 circular IoU/中心差/面积比，以及跨 seam、高纬度、大 FOV 和快速尺度变化分层结果。

## 6. 实验四：TRACKING/UNCERTAIN 自适应 ViewSpecType1

### 6.1 目的和对照

只改变 Type1 实际 FOV 和中心偏移，视图数仍为 `4+4`。动态尺寸来自 `predictedMotion`，无合法尺寸时回退上一可信 BFoV 尺寸，再无效时使用固定 `120°`。

生产对照保持当前行为：TRACKING 两轮动态 Type1，UNCERTAIN 两轮固定 `120°` Type1。

### 6.2 变体

- `fov_adaptive_both_rounds`：TRACKING 和 UNCERTAIN 的两轮都使用动态 Type1。
- `fov_adaptive_round1_only`：两种状态的 Round 1 使用动态 Type1，Round 2 固定 `120° x 120°` Type1。

不得改用第一轮候选尺寸替换本实验规定的运动预测尺寸，否则会同时改变尺寸估计语义。

### 6.3 专用记录

记录按状态/轮次的 FOV 分布、上下限触发率、R1/R2 真值覆盖率、目标在局部图中的位置和面积、第一轮到第二轮的中心误差改善，以及按目标大小、速度、纬度、seam 和 loss episode 的分层指标。

## 7. 实验五：IoU/BBox Refinement

### 7.1 目的和隔离

只增加一次局部 bbox refinement，不改变搜索中心、视图数量、候选排序、Fusor、状态机或模板。训练使用 `train` split，validation 只用于本实验的最终比较，holdout 禁止读取。新 checkpoint 必须在独立 calibration split 重新校准。

### 7.2 变体

`iou_refine_head` 使用 HiT 现有特征、corner 输出和归一化初始 local bbox 预测 `dx/dy/log-w/log-h`。修正框裁剪到 LocalView 后按生产 Geometry 回投。输出非有限、越界、无效或 gate 失败时回退未修正框并计数。

### 7.3 专用记录

记录初始框和 refined 框的局部/ERP/BFoV 指标、触发率、回退率、中心/宽高/面积改善、接受率变化和各分层结果，特别比较 `seq_0017` 与 `seq_0045`。

## 8. 实验六：Distractor-aware Identity Verifier

### 8.1 目的和固定边界

只增加候选身份一致性验证，不改变模板内容、视图计划或生产 Fusor。prototype 只能来自 `valid=true` 且通过 measurement acceptance 的正式结果，不能使用真值 IoU、预测框或第一轮未提交观测。

### 8.2 变体

`distractor_identity_verifier` 同时计算候选与 frame 0 anchor、最近稳定 prototype 的相似度、prototype 一致性和运动/尺度创新量，再对 SingleScore 做单调门控或 rerank。prototype 不可用、输出非有限或验证器失败时回退生产排序。

验证器不得直接触发模板更新。记录高分低 IoU 候选的拒绝率、身份切换、错误目标持续长度、恢复帧数、相似度分布和 distractor/遮挡/seam/FOV 分层结果。

## 9. 实验七：局部跟踪与全局恢复双候选

### 9.1 目的和 LostReport 对齐

`LostReport.md` 已证明直接切换 LOST 会破坏 `seq_0017` 上仍然有效的局部链。本实验保持局部 proposal 与全局 recovery proposal 并行存在，只有恢复候选经过因果确认后才替换公开结果。它是独立的隔离实验，不修改正常状态机的默认行为。

全局 proposal 固定使用 `cube6_adaptive_type1`；本实验变量是局部/全局双候选与提交策略，不重新比较 LOST ViewSpec。

### 9.2 变体

`local_global_recovery_verifier` 使用连续低 SingleScore、无有效候选、运动创新过大、cornerScore 下降等运行时可观测信号触发 shadow recovery：

1. 保留局部 proposal、运动预测和已提交历史。
2. 以预测中心执行全局 recovery。
3. 分别计算 appearance、identity、运动一致性、尺度跳变和来源可靠性。
4. recovery 连续两帧通过 acceptance gate 且与 anchor identity 一致后，才从第二个确认帧开始替换公开结果。
5. 确认失败时取消 pending recovery，继续局部链；成功替换后调用 `resetFromMeasurement()` 清除旧速度。

不得回溯已提交帧，不得用真值 IoU 触发或确认，不得因 recovery 失败清空局部结果。

### 9.3 专用记录

记录 shadow trigger、recovery proposal、确认成功率、确认延迟、局部/全局选择比例、错误替换、恢复后再次丢失、恢复持续长度，以及 `seq_0017` 局部链保护和 `seq_0045` absent 区间的结果。

## 10. 实验八：流水线与 GPU Geometry 裁图

### 10.1 目的和实验边界

本实验集专门测量两类执行优化及其交互：

1. 流水线是否能在保持帧事务、状态提交和两轮依赖语义的前提下，隐藏 decode、准备和推理之间的等待。
2. Geometry 的 ERP/透视 crop、resize 和预处理搬到 GPU 后，是否能减少 CPU 开销和数据搬运。
3. 两者同时实现时，是否能形成真实的端到端收益，而不是把两个局部收益相加。

本实验集只有一个共享生产对照和三个独立实验组。三个组都必须从相同的共享对照开始，不能先运行一个组再把它的输出接到另一个组。第三组是有意测试两种优化交互的组合实验，不把它解释成单变量结果。

### 10.2 共享对照：`shared_control_production`

使用当前生产顺序线程、CPU Geometry crop、CPU preprocessing、每帧顺序 `4+4`、每轮一次 HiT GPU forward。该对照只需生成一次，所有比较文件必须记录其 artifact hash。

### 10.3 共同 GPU Geometry 语义

GPU Geometry 变体必须满足以下数据流：

```text
CPU FramePacket
  -> 一次必要的 H2D
  -> GPU Geometry crop / resize / color conversion / preprocess
  -> GPU HiT tensor
  -> 下一轮 GPU HiT tensor 或当前轮后续 GPU 计算
```

- crop、resize、颜色通道处理和预处理在 GPU 上完成，输出直接是 HiT 所需的 device tensor。
- 不得把 LocalView RGB 转回 CPU `numpy`、PIL 或 `uint8`，再转回 GPU。
- 不得在两轮之间对 crop 图像执行 `GPU -> CPU -> uint8/float32 -> GPU` 往返。
- Round 1 结果若需要由 CPU Controller 决定 Round 2 中心，只允许传输必要的有限标量/候选元数据；禁止传输 crop 图像或重复转换图像 tensor。
- Round 2 的 Geometry 输出必须直接进入下一次 HiT GPU forward。GPU tensor 的 dtype 必须与当前 FP32 HiT 合同一致；本实验不得顺便启用 FP16、TensorRT 或改变模型输入尺寸。
- GPU crop 与 CPU crop 必须先做像素级和边界回归，覆盖 seam、极点、FOV 上下限、padding、颜色通道、空视图和非连续 batch。
- 所有 CUDA stream/event 依赖必须显式记录，不能用隐含全局同步掩盖数据竞争。

这里的“直接进入下一轮 HiT GPU 计算”指图像数据保持在 GPU 上直接被下一次模型推理消费；候选中心、bbox、valid 等控制标量是否需要短暂返回 CPU，由现有 Controller 协议决定，但不得把它扩大成图像数据往返。

### 10.4 实验组一：`pipeline_only`

只启用流水线，不启用 GPU Geometry。Geometry crop、resize 和 preprocessing 仍走当前 CPU 路径。

流水线至少覆盖：

- frame `n+1` 的读取、解码和 CPU 准备与 frame `n` 的 GPU HiT/Controller 处理重叠；
- frame 内 Round 2 仍必须等待 Round 1 Fusor 中心，不能用未经确认的未来状态破坏两轮语义；
- FrameTransaction 只能按帧号顺序提交一次；
- generation/revision、乱序、迟到结果、异常、OOM、sequence close 和取消都必须有测试；
- stale result 只能丢弃，不能覆盖较新的 Controller、运动历史或模板 revision。

该组只测调度收益和调度开销，不得偷偷启用 GPU crop。

### 10.5 实验组二：`gpu_geometry_only`

只启用 GPU Geometry，不启用跨帧流水线。执行仍保持顺序，但两轮的 crop/resize/preprocess 输出直接交给 HiT GPU。

必须分别记录：

- 一次 H2D 后的 GPU 图像生命周期；
- Round 1 和 Round 2 的 GPU crop、resize、preprocess、HiT forward；
- CPU round-trip 图像次数，目标为 `0`；
- 图像 dtype/device 变化次数，目标为只发生生产契约要求的初始化转换；
- CPU/GPU 同步点、GPU stream 等待和额外显存；
- GPU/CPU crop 的像素差异、候选差异和最终 TrackResult 差异。

如果 GPU crop 的数值差异改变模型结果，必须同时报告“像素级通过但跟踪指标变化”和“数值/边界回归失败”两种情况，不能只用端到端均值掩盖 Geometry 回归。

### 10.6 实验组三：`pipeline_gpu_geometry`

同时启用跨帧流水线和 GPU Geometry。该组必须复用前两组相同的输入、模型、数据和对照，但实现上仍创建全新的 Runtime、队列和 CUDA stream。

除了共同记录外，必须确认：

- CPU 解码、H2D、GPU crop、Round 1 HiT、Round 2 GPU crop 和 Round 2 HiT 的依赖关系正确；
- GPU crop 生成的 tensor 不因流水线队列而落回 CPU；
- 不同帧之间没有复用错误的 frame buffer、crop tensor、template revision 或 stream event；
- 迟到的 frame `n` 结果不能覆盖已经提交的 frame `n+1`；
- sequence close、OOM、异常和取消后所有 GPU tensor、stream、worker 和队列都能关闭；
- 吞吐收益不能以增加 stale drop、回退、缺帧或状态错序为代价。

### 10.7 实验八专用指标

除共同指标外，实验八必须报告：

- CPU crop、GPU crop、resize、preprocess、H2D、D2H、HiT forward 和 total 的 P50/P95/P99；
- 每帧 CPU/GPU round-trip 次数、图像 tensor 转换次数、控制标量传输次数；
- GPU tensor 的 dtype、device、shape、stream 和 event；
- crop 像素最大误差、P99 误差、边界坐标误差和颜色通道差异；
- pipeline queue wait、stage overlap、in-flight 帧数、stale drop、回退、重试和取消；
- 平均视图数、forward 数、吞吐、峰值显存、GPU 利用率、温度和 P95/P99 长尾；
- 与共享对照逐帧 `TrackResult`、候选排序、状态、模板 revision 和运动历史的一致性。

### 10.8 实验八短测

正式 validation 前，三个变体分别通过：

1. CPU/GPU Geometry 对同一 ViewSpec 的像素和边界回归。
2. 单帧、双轮、跨 seam、极点、padding 和无效 ViewSpec 回退。
3. pipeline 的乱序、迟到、OOM、异常、取消、sequence close 和重复提交测试。
4. GPU tensor 保持 device 侧并直接进入下一轮 HiT 的断言。
5. 人工中断和 resume 后无重复提交、无残留 stream、无旧 revision 复用。

## 11. 统一比较与判定

每个实验集完成后单独生成：

```text
artifacts/trying_experiments/
  <experiment>/
    control/
    <variant>/
      <sequence>/
      aggregate.json
      per_sequence.csv
      comparison.md
      decision.json
```

实验集之间不共享运行产物，除了经过 hash 校验的 `control`。每个实验集都必须先检查任务完整性、逐帧对齐、非有限值、OOM、环境和 hash，再比较精度和性能。

默认通过条件：

- overall mean IoU 和 spherical IoU 不明显下降；
- success@0.5 不下降超过 `0.5` 个百分点；
- tracking loss rate 和 absent FPR 不恶化超过 `1.0` 个百分点；
- `seq_0017` 的 mean IoU 和 loss rate 不差于匹配对照；
- `seq_0045` 的 absent 区间不出现未解释的错误输出；
- 不新增非有限输出、未处理 OOM、缺帧、状态错序或模板串序列；
- 性能优化必须在阶段耗时和端到端 P95/P99 中有可重复收益，不能只降低某个未计入总时间的局部计时；
- GPU Geometry 必须证明没有图像 CPU 往返，且不能用改变模型精度来制造收益；
- pipeline 必须证明提交顺序、revision、状态和结果语义与顺序对照一致。

若某个实验集没有变体同时满足完整性、逐序列精度和性能条件，则只保留共享 Stage 3 FP32 顺序 `4+4` 对照，并在本文件开头登记每个变体的失败原因。一个实验集失败不影响其他实验集独立运行和判定。

## 12. 结果冻结规则

本计划只产生隔离实验结果，不直接修改生产默认开关。任何候选进入生产前，必须把实现、配置、测试、产物和回退路径另行冻结，并再次执行针对性回归。

所有候选失败也必须保留实验产物和失败原因。最终 holdout 只能在模型、calibration、Controller、Geometry 和采用的执行路径全部冻结后读取一次；不得用 holdout 选择本计划中的任一实验变体。
