# InstaTargetingSystem 超参数索引

> 本文档登记当前配置文件和预留设计中会改变模型选择、计算精度、几何视野、深度门控、决策、恢复或运行资源的全部配置项。
> 行号以当前仓库版本为准；配置结构调整后必须同步更新。实验记录中的指标建议至少填写 `AUC`、`Success Rate@0.5` 和 `FPS`。

---

## 1. 登记范围

以下内容计入本索引：

- YAML 中影响算法、模型或运行资源的配置项；
- 代码中影响数值判定、且不能由 YAML 注入的数值容差。

以下内容不属于超参数：

- `schemaVersion` 等格式版本；
- 概率必须位于 `[0, 1]`、尺寸必须为正等协议不变量；
- 测试数据、依赖版本和文档示例值。

配置加载后，`minFovDeg` 和 `maxFovDeg` 会立即转换为弧度。相对权重路径以配置文件目录为基准解析。
几何层的边界采样预算由 `geometry.boundarySamplesPerEdge` 控制。

---

## 2. `configs/RGBD.yaml`

该文件是未来 RGB-D 形态的主配置草案；当前第三阶段主要以 `configs/RGBonly.yaml` 作为运行配置。

| 参数名 | 当前行 | 当前值 | 简要职能 | 调大或切换后的主要影响 |
|---|---:|---|---|---|
| `model.backend` | 3 | `pytorch` | 选择推理运行时 | 切换后影响部署兼容性、速度与数值一致性 |
| `model.variant` | 4 | `hit_small` | 选择 HiT 模型规模 | 更大变体通常提高容量，同时降低速度并增加显存 |
| `model.weights` | 5 | `../models/hit_small.pth` | 指定模型权重 | 切换权重会直接改变精度、泛化和可复现性 |
| `model.precision` | 6 | `fp16` | 指定推理数值精度 | 更低精度通常提高速度、降低显存，但可能增加临界分数误差 |
| `geometry.viewWidthPx` | 8 | `256` | 设置局部视图宽度 | 调大可保留更多细节，但提高投影和推理成本 |
| `geometry.viewHeightPx` | 9 | `256` | 设置局部视图高度 | 调大可保留更多垂直细节，但提高投影和推理成本 |
| `geometry.boundarySamplesPerEdge` | 10 | `65` | 控制每条边界的采样预算 | 调大可提高极区与跨经线边界的包络精度，但增加几何计算成本 |
| `geometry.minFovDeg` | 11 | `20.0` | 限制最小搜索视场 | 调大可覆盖更大位移，但目标像素占比下降 |
| `geometry.maxFovDeg` | 12 | `120.0` | 限制最大搜索视场 | 调大有利于恢复大位移目标，但畸变、误匹配和成本上升 |
| `depth.enabled` | 14 | `true` | 未来完整深度链路开关 | 当前第三阶段不生效；第四阶段开启后才增加计算和数据依赖 |
| `depth.minValidRatio` | 15 | `0.35` | 未来深度摘要最小有效像素比例 | 当前第三阶段不参与打分；第四阶段调大可提高深度可靠性 |
| `depth.maxDepthJumpRatio` | 16 | `0.60` | 未来允许的相邻深度跳变比例 | 当前第三阶段不参与打分；第四阶段调大可容忍快速距离变化 |
| `backendFusion.depthScoreWeight` | 18 | `0.15` | 未来深度分数在后端融合中的权重 | 当前第三阶段不生效；第四阶段调大增强深度影响，也会放大深度噪声风险 |
| `decisionGate.motionScoreWeight` | 20 | `0.25` | 控制运动连续性门控权重 | 调大更偏好预测轨迹附近候选，可能抑制突变运动 |
| `decisionGate.scaleScoreWeight` | 21 | `0.15` | 控制尺度连续性门控权重 | 调大更排斥尺度突变，也可能错过快速接近目标 |
| `tracking.acceptThreshold` | 23 | `0.70` | 直接接受观测的最低置信度 | 调大可减少误接受，但会增加不确定和恢复状态 |
| `tracking.uncertainThreshold` | 24 | `0.45` | 区分不确定与低置信观测 | 调大更早触发恢复，成本和找回机会同时增加 |
| `tracking.stableFramesBeforeUpdate` | 25 | `8` | 更新稳定模板前要求的连续稳定帧数 | 调大可降低模板污染，但外观适应速度变慢 |
| `tracking.windowLength` | 26 | `5` | 多帧运动预测使用的历史窗口长度 | 调大可平滑噪声，但增加滞后并减弱急转响应 |
| `recovery.maxViewsPerFrame` | 28 | `12` | 限制每帧恢复候选视图数 | 调大提高搜索覆盖率，同时增加延迟与显存占用 |
| `recovery.globalSearchInterval` | 29 | `5` | 控制全景粗搜的帧间隔 | 调大降低恢复成本，但目标找回可能更慢 |
| `runtime.decodeQueueCapacity` | 31 | `3` | 解码到控制线程的队列容量 | 调大可吸收短时抖动，但增加全景帧内存占用 |
| `runtime.inferRequestQueueCapacity` | 32 | `1` | 控制到推理线程的请求队列容量 | 调大增加排队和状态过期风险，通常保持单槽 |
| `runtime.inferResponseQueueCapacity` | 33 | `1` | 推理到控制线程的响应队列容量 | 调大增加未提交响应驻留，通常保持单槽 |
| `runtime.resultQueueCapacity` | 34 | `32` | 控制到结果线程的队列容量 | 调大可隔离输出抖动，但增加结果驻留内存 |

### 2.1 效果记录表

每次只改变一个参数；若必须联调多个参数，在备注中列出完整组合。

| 参数名 | 日期/实验 ID | 原值 | 新值 | 数据集/序列 | AUC | SR@0.5 | FPS | 找回成功率 | 现象与结论 |
|---|---|---|---|---|---:|---:|---:|---:|---|
| `model.backend` |  |  |  |  |  |  |  |  |  |
| `model.variant` |  |  |  |  |  |  |  |  |  |
| `model.weights` |  |  |  |  |  |  |  |  |  |
| `model.precision` |  |  |  |  |  |  |  |  |  |
| `geometry.viewWidthPx` |  |  |  |  |  |  |  |  |  |
| `geometry.viewHeightPx` |  |  |  |  |  |  |  |  |  |
| `geometry.boundarySamplesPerEdge` |  |  |  |  |  |  |  |  |  |
| `geometry.minFovDeg` |  |  |  |  |  |  |  |  |  |
| `geometry.maxFovDeg` |  |  |  |  |  |  |  |  |  |
| `depth.enabled` |  |  |  |  |  |  |  |  |  |
| `depth.minValidRatio` |  |  |  |  |  |  |  |  |  |
| `depth.maxDepthJumpRatio` |  |  |  |  |  |  |  |  |  |
| `backendFusion.depthScoreWeight` |  |  |  |  |  |  |  |  |  |
| `decisionGate.motionScoreWeight` |  |  |  |  |  |  |  |  |  |
| `decisionGate.scaleScoreWeight` |  |  |  |  |  |  |  |  |  |
| `tracking.acceptThreshold` |  |  |  |  |  |  |  |  |  |
| `tracking.uncertainThreshold` |  |  |  |  |  |  |  |  |  |
| `tracking.stableFramesBeforeUpdate` |  |  |  |  |  |  |  |  |  |
| `tracking.windowLength` |  |  |  |  |  |  |  |  |  |
| `recovery.maxViewsPerFrame` |  |  |  |  |  |  |  |  |  |
| `recovery.globalSearchInterval` |  |  |  |  |  |  |  |  |  |
| `runtime.decodeQueueCapacity` |  |  |  |  |  |  |  |  |  |
| `runtime.inferRequestQueueCapacity` |  |  |  |  |  |  |  |  |  |
| `runtime.inferResponseQueueCapacity` |  |  |  |  |  |  |  |  |  |
| `runtime.resultQueueCapacity` |  |  |  |  |  |  |  |  |  |

---

## 3. `configs/RGBonly.yaml`

该文件是当前第三阶段的 RGB-only 运行配置。参数职责与 RGB-D 配置相同，差异值重点用于对照实验。

| 参数名 | 当前行 | 当前值 | 简要职能 | 与 RGB-D 主配置的差异 |
|---|---:|---|---|---|
| `model.backend` | 3 | `pytorch` | 选择推理运行时 | 相同 |
| `model.variant` | 4 | `hit_small` | 选择 HiT 模型规模 | 相同 |
| `model.weights` | 5 | `../models/hit_small.pth` | 指定模型权重 | 相同 |
| `model.precision` | 6 | `fp32` | 指定推理数值精度 | 使用 FP32 作为 RGB-only 参考精度 |
| `geometry.viewWidthPx` | 8 | `256` | 设置局部视图宽度 | 相同 |
| `geometry.viewHeightPx` | 9 | `256` | 设置局部视图高度 | 相同 |
| `geometry.boundarySamplesPerEdge` | 10 | `65` | 控制每条边界的采样预算 | 相同 |
| `geometry.minFovDeg` | 11 | `20.0` | 限制最小搜索视场 | 相同 |
| `geometry.maxFovDeg` | 12 | `120.0` | 限制最大搜索视场 | 相同 |
| `depth.enabled` | 14 | `false` | 深度链路预留开关 | 当前第三阶段保持关闭 |
| `depth.minValidRatio` | 15 | `0.35` | 未来深度摘要最小有效像素比例 | 当前后端不读取；保留用于后续切换配置 |
| `depth.maxDepthJumpRatio` | 16 | `0.60` | 未来允许的相邻深度跳变比例 | 当前后端不读取；保留用于后续切换配置 |
| `backendFusion.depthScoreWeight` | 18 | `0.0` | 未来深度分数融合权重 | 当前 RGB-only 后端固定为零 |
| `decisionGate.motionScoreWeight` | 20 | `0.25` | 控制运动连续性门控权重 | 相同 |
| `decisionGate.scaleScoreWeight` | 21 | `0.15` | 控制尺度连续性门控权重 | 相同 |
| `tracking.acceptThreshold` | 23 | `0.70` | 直接接受观测的最低置信度 | 相同 |
| `tracking.uncertainThreshold` | 24 | `0.45` | 区分不确定与低置信观测 | 相同 |
| `tracking.stableFramesBeforeUpdate` | 25 | `8` | 更新稳定模板前要求的连续稳定帧数 | 相同 |
| `tracking.windowLength` | 26 | `5` | 多帧运动预测使用的历史窗口长度 | 相同 |
| `recovery.maxViewsPerFrame` | 28 | `12` | 限制每帧恢复候选视图数 | 相同 |
| `recovery.globalSearchInterval` | 29 | `5` | 控制全景粗搜的帧间隔 | 相同 |
| `runtime.decodeQueueCapacity` | 31 | `3` | 解码到控制线程的队列容量 | 相同 |
| `runtime.inferRequestQueueCapacity` | 32 | `1` | 控制到推理线程的请求队列容量 | 相同 |
| `runtime.inferResponseQueueCapacity` | 33 | `1` | 推理到控制线程的响应队列容量 | 相同 |
| `runtime.resultQueueCapacity` | 34 | `32` | 控制到结果线程的队列容量 | 相同 |

### 3.1 效果记录表

| 参数名 | 日期/实验 ID | 原值 | 新值 | 数据集/序列 | AUC | SR@0.5 | FPS | 找回成功率 | 现象与结论 |
|---|---|---|---|---|---:|---:|---:|---:|---|
| `model.backend` |  |  |  |  |  |  |  |  |  |
| `model.variant` |  |  |  |  |  |  |  |  |  |
| `model.weights` |  |  |  |  |  |  |  |  |  |
| `model.precision` |  |  |  |  |  |  |  |  |  |
| `geometry.viewWidthPx` |  |  |  |  |  |  |  |  |  |
| `geometry.viewHeightPx` |  |  |  |  |  |  |  |  |  |
| `geometry.minFovDeg` |  |  |  |  |  |  |  |  |  |
| `geometry.maxFovDeg` |  |  |  |  |  |  |  |  |  |
| `depth.enabled` |  |  |  |  |  |  |  |  |  |
| `depth.minValidRatio` |  |  |  |  |  |  |  |  |  |
| `depth.maxDepthJumpRatio` |  |  |  |  |  |  |  |  |  |
| `backendFusion.depthScoreWeight` |  |  |  |  |  |  |  |  |  |
| `decisionGate.motionScoreWeight` |  |  |  |  |  |  |  |  |  |
| `decisionGate.scaleScoreWeight` |  |  |  |  |  |  |  |  |  |
| `tracking.acceptThreshold` |  |  |  |  |  |  |  |  |  |
| `tracking.uncertainThreshold` |  |  |  |  |  |  |  |  |  |
| `tracking.stableFramesBeforeUpdate` |  |  |  |  |  |  |  |  |  |
| `tracking.windowLength` |  |  |  |  |  |  |  |  |  |
| `recovery.maxViewsPerFrame` |  |  |  |  |  |  |  |  |  |
| `recovery.globalSearchInterval` |  |  |  |  |  |  |  |  |  |
| `runtime.decodeQueueCapacity` |  |  |  |  |  |  |  |  |  |
| `runtime.inferRequestQueueCapacity` |  |  |  |  |  |  |  |  |  |
| `runtime.inferResponseQueueCapacity` |  |  |  |  |  |  |  |  |  |
| `runtime.resultQueueCapacity` |  |  |  |  |  |  |  |  |  |

---

## 4. `src/instatarget/core/types.py`

该文件没有算法超参数，只有一个影响协议数值判定的固定容差。它不进入 YAML，原因是所有模块必须在加载应用配置前也能构造和校验核心类型。

| 参数名 | 当前行 | 当前值 | 简要职能 | 调大后的主要影响 |
|---|---:|---:|---|---|
| `UNIT_VECTOR_TOLERANCE` | 18 | `1e-6` | 判断 `SphericalPoint.xyz` 是否可视为单位向量 | 可容忍更大的归一化误差，但会放宽球面坐标不变式 |

### 4.1 效果记录表

| 参数名 | 日期/实验 ID | 原值 | 新值 | 触发样本 | 误拒绝数 | 误接受数 | 数值现象与结论 |
|---|---|---|---|---|---:|---:|---|
| `UNIT_VECTOR_TOLERANCE` |  |  |  |  |  |  |  |

---

## 5. 维护规则

1. 新增或修改超参数时，先修改对应 YAML，再更新本文件的行号、当前值和职能。
2. 算法代码不得新增未登记的阈值、权重、窗口长度或恢复预算。
3. 对照实验应固定代码提交、权重哈希、数据集划分和随机种子。
4. 参数联调必须在备注中记录完整组合，避免把联合效果错误归因给单个参数。
5. 配置文件移动或插入行后，应重新核对“当前行”，禁止保留过期行号。

---

## 6. 第三阶段 tracker 后端登记

第三阶段只实现 RGB-only 后端协议、HiT 会话适配、观测规范化和模板命令执行，未新增可调超参数。
HiT 的模型、权重、精度仍由现有 `model.*` 参数控制；RGB-only 的 `depthScore=0` 和
`fusedScore=appearanceScore` 是模式契约，不是可调权重。

以下固定行为也不计入超参数：锚点/近期/稳定三个模板槽位、模板特征传递顺序、命令 revision
校验规则、局部框裁剪规则以及 `latencyNs` 的单调时钟来源。任何将来改变这些行为的数值配置，
必须先新增 YAML 字段，再在本文档登记。

---

## 7. 第四阶段深度颜色化与双 HiT 融合

以下条目已进入 `configs/RGBD.yaml` 和 `configs/RGBonly.yaml`。RGB-only 配置仍关闭深度链路，
但保留相同字段以保证模式切换时配置结构稳定。

| 参数名 | 建议默认值 | 简要职能 | 备注 |
|---|---:|---|---|
| `depth.colorization.mode` | `relief` | 选择深度到颜色的编码方式 | 推荐浮雕式单调映射 |
| `depth.colorization.nearBrightness` | `0.95` | 近处亮度上限 | 站立前景更亮 |
| `depth.colorization.farBrightness` | `0.20` | 远处亮度下限 | 地面与背景更暗 |
| `depth.colorization.reliefGain` | `1.00` | 浮雕起伏增益 | 控制前景“凸出”强度 |
| `depth.colorization.edgeGain` | `0.35` | 轮廓增强增益 | 控制边缘锐化程度 |
| `depth.colorization.smoothingKernel` | `7` | 背景平滑核尺寸 | 估计地面/背景平面 |
| `fusionHead.rgbInitWeight` | `0.70` | 融合头初始 RGB 权重 | 初值偏向主视觉分支 |
| `fusionHead.depthInitWeight` | `0.20` | 融合头初始深度权重 | 深度作为辅助判别 |
| `fusionHead.contextInitWeight` | `0.10` | 融合头初始上下文权重 | 模板 / 运动 / 尺度残差 |

### 7.1 设计说明

- 这些参数只服务于第四阶段；关闭深度链路时不会改变 RGB-only 行为。
- `depth.colorization.*` 用于把深度图转成更容易被 HiT 分辨轮廓的伪彩色图。
- `fusionHead.*` 只用于 MLP 融合头的初始值和训练起点，不代表最终固定权重。
- 若后续选择改成显式深度编码器，也应保留同一组接口命名，以免文档和实验记录断裂。
