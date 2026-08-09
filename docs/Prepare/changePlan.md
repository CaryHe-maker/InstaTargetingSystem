# InstaTargetingSystem 修改计划

> 本文档仅覆盖 `changeRecommend.md` 中的问题 1、2、3、4、5、6。  
> 目标是把项目整理成一条可直接实现的统一链路，并作为后续改代码的执行模板。
> 这是一份历史提案。前四阶段的实现状态以 `docs/implement.md`、`docs/interface.md` 和
> `docs/modules/tracker.md` 为准；其中 RGB-D 深度伪彩色、深度分支和融合头已经落地。
> 第五阶段 DTC 的现行蓝图以 `docs/ReplySugg.md` 和 `docs/modules/controller.md` 为准，本文
> 中与“DTC 直接读取 BFoV 深度块”“单视图正常跟踪”冲突的描述均已废止。

---

## 0. 最终目标

本项目最终采用以下结构：

- `geometry` 只做 RGB / Depth 同步裁剪和几何对齐。
- `TrackerBackend` 统一完成深度图像处理、HiT 跟踪和 MLP 融合。
- `DTC` 只做状态维护、多帧预测、搜索计划生成和最终门控。
- 同一套代码同时支持 `rgb_depth` 和 `rgb_only`。

核心约束：

1. RGB 和 Depth 必须同路传递。
2. 后端内部完成深度图处理、局部跟踪和融合。
3. 控制层不再实现深度神经网络或融合头。
4. 预测和搜索计划由 `DTC` 收口。

---

## 1. 修改原则

| 原则 | 要求 |
|------|------|
| 输入同路 | 只要路径传递 RGB，就必须同时携带对应深度或明确标记为 `None` |
| 几何只对齐 | `geometry` 负责同视角裁剪、回投影和畸变控制，不做状态判断 |
| 控制层主导 | 运动方向、速度、下一帧搜索中心、搜索视场都由 `DTC` 决定 |
| 后端单责 | `TrackerBackend` 负责深度预处理、局部跟踪和融合，不做全局恢复和状态机决策 |
| 深度可降级 | 深度缺失时，系统必须自动退化为 `rgb_only` |
| 主干可复用 | 现有 HiT 预训练权重保留，只训练后端新增深度模块和融合头 |

---

## 2. 总体架构

### 2.1 数据流

`FramePacket -> geometry -> DTC -> SearchPlan -> TrackerBackend -> ProjectedObservation -> DTC`

### 2.2 模块职责

| 模块 | 职责 |
|------|------|
| `data` | 读取 `rgb_depth` / `rgb_only` 序列 |
| `geometry` | 同步裁剪 RGB 和 Depth，生成局部视图 |
| `DTC` | 维护状态、预测下一帧位置、生成搜索计划 |
| `TrackerBackend` | 执行深度预处理、局部跟踪和 MLP 融合，只输出局部观测 |
| `fusion` | 候选排序与最终门控 |

---

## 3. 问题 1：`geometry` 同步处理颜色图和深度图

### 3.1 需要修改的模块

| 位置 | 修改内容 |
|------|----------|
| `docs/interface.md` | 让 `FramePacket`、`LocalView`、`ViewSpec` 明确携带深度或缺省深度 |
| `docs/modules/geometry.md` | 规定几何层必须输出 RGB 和深度的同视角裁剪结果 |
| `src/instatarget/geometry/` | 实现 RGB / Depth 同步裁剪、回投影和视场生成 |
| `src/instatarget/core/` | 增加深度类型、视场锚点类型、深度缺失标记 |

### 3.2 实际实现要求

1. `FramePacket` 必须同时持有 `rgb` 与 `depth`。
2. `LocalView` 必须同时持有 `rgb` 与 `depth`，两者共享同一个 `ViewSpec`。
3. `cropViews()` 必须按同一条 BFoV 采样线同时裁出颜色图和深度图。
4. `bfovToBbox()` 和 `localBoxToBfov()` 只能做几何转换，不能丢弃深度上下文。
5. `geometry` 只返回“对齐后的图”，不解释深度语义。

### 3.3 算法要求

- 默认以 `DTC` 提供的搜索中心作为 BFoV 中心。
- 初始化阶段允许由首帧框生成模板视图。
- 若只有一个点，`geometry` 用该点作为视场中心，FOV 尺寸由 `ViewSpec` 给出。
- 回投影时使用边界采样，不只采四个角。
- 为避免形变，局部视图统一以 BFoV 为中介，不直接从 ERP 框硬切。

### 3.4 完成条件

- 任一带深度输入的序列，`geometry` 输出的局部图都能同时拿到 RGB 和深度。
- 任一不带深度的序列，`geometry` 仍能正常输出 RGB，深度字段为 `None`。

---

## 4. 问题 2：`DTC` 读取 BFoV 区域深度并进行多帧预测

### 4.1 需要修改的模块

| 位置 | 修改内容 |
|------|----------|
| `docs/modules/controller.md` | 增加 DTC 的具体运动估计算法 |
| `docs/interface.md` | 增加 `DepthSummary`、`DepthPatchSummary`、`MotionState3D`、`predictedMotion` |
| `src/instatarget/controller/` | 实现 DTC、运动预测、搜索计划生成 |
| `src/instatarget/core/` | 增加运动状态、深度状态、预测状态类型 |

### 4.2 默认算法

`DTC` 默认采用滑动窗口式球面方向 + 深度距离轻量预测模型。  
不是只用一帧，而是使用前 `n` 帧连续观测一起做路径规划，`n` 为配置项，默认 3 到 5。

1. 收集最近 `n` 帧的目标中心、深度摘要、BFoV 深度块和置信度。
2. 把每帧球面方向转成单位向量序列。
3. 把每帧深度转成相对距离序列。
4. 用窗口内方向变化、深度变化和角速度估计当前趋势。
5. 用轻量常速度模型或 Alpha-Beta / Kalman 滤波预测下一帧中心。
6. 用窗口内深度跳变和角速度共同决定下一帧搜索中心和搜索视场。
7. 把预测结果转成下一帧 `SearchPlan`。

### 4.3 窗口内必须消费的数据

| 数据 | 来源 | 用途 |
|------|------|------|
| 最近 `n` 帧球面中心 | 历史 `TrackResult` | 估计方向趋势和角速度 |
| 最近 `n` 帧深度摘要 | 历史 `DepthSummary` | 估计距离变化和遮挡趋势 |
| 最近 `n` 帧 BFoV 深度块 | 历史局部视图 | 估计边界、前景/背景分离和恢复方向 |
| 最近 `n` 帧置信度 | 历史观测 | 判断是否稳定、是否该更新模板 |
| 360VOT 方向变化率 | 球面中心变化 | 修正全景相机运动带来的偏差 |

### 4.4 预测输出

`DTC` 每帧必须产出：

- `predictedMotion`
- `next search center`
- `next search FOV`
- `templateCommand`
- `TrackResult`

### 4.5 完成条件

- `DTC` 不再只读深度摘要。
- `DTC` 能利用最近 `n` 帧的 BFoV 内深度和前后帧深度变化做预测。
- `DTC` 能在无深度时自动退化为 RGB + 球面几何模式。

---

## 5. 问题 4：深度和颜色处于同一层级

### 5.1 需要修改的模块

| 位置 | 修改内容 |
|------|----------|
| `docs/interface.md` | 所有传 RGB 的接口都要同步保留深度字段 |
| `docs/modules/tracker.md` | 后端输入视图保留 `depth` 字段，后端内部完成深度预处理和融合 |
| `docs/modules/geometry.md` | 几何层输出 RGB / Depth 成对结果 |
| `src/instatarget/app/` | 运行时只传单独 RGB 的路径改成成对传递 |

### 5.2 接口规则

1. 颜色图和深度图必须共享同一 `FramePacket`。
2. 几何层输出的 `LocalView` 必须保持颜色和深度对齐。
3. `TrackerBackend` 的输入对象必须携带深度字段。
4. 后端内部完成深度图处理、HiT 和 MLP 融合。
5. `geometry -> DTC` 与 `geometry -> TrackerBackend` 的路径都必须保留深度。

### 5.3 深度与颜色的关系

- 颜色负责外观区分。
- 深度负责空间区分。
- 两者都参与候选筛选。
- 颜色相近但深度差异大时，优先保留深度一致的候选。
- 深度缺失时，颜色路径继续工作，深度权重归零。

### 5.4 完成条件

- 所有 RGB 传递路径都能找到对应深度通道。
- 后端接口不再把深度当成“丢掉的附加信息”。

---

## 6. 问题 5：同时支持有深度和无深度的数据集

### 6.1 需要修改的模块

| 位置 | 修改内容 |
|------|----------|
| `docs/data.md` | 定义 `rgb+depth` 和 `rgb-only` 两种数据模式 |
| `docs/interface.md` | 给 `FramePacket.depth`、`LocalView.depth`、`DepthSummary` 加空值语义 |
| `docs/modules/controller.md` | 规定深度缺失时的退化策略 |
| `src/instatarget/data/` | 读取序列时自动识别是否存在深度 |
| `src/instatarget/controller/` | 深度缺失时自动把深度权重降为 0 |
| `tests/` | 增加有深度和无深度两套回归样例 |

### 6.2 数据模式

| 模式 | 输入 | 行为 |
|------|------|------|
| `rgb_depth` | 颜色图 + 深度图 | 启用完整深度链路 |
| `rgb_only` | 只有颜色图 | 退化为 RGB + 球面几何 |

### 6.3 退化规则

1. 如果序列没有深度，`FramePacket.depth = None`。
2. `DepthProcessor` 返回 `None`。
3. `DTC` 把深度权重设为 0。
4. `TrackerBackend` 正常运行，只依赖 RGB 视图。
5. 输出格式不变。

### 6.4 完成条件

- 同一套代码能跑 `rgb_depth` 和 `rgb_only`。
- 无深度数据集不会引发空指针或缺字段错误。

---

## 7. 问题 3：颜色与深度如何共同送入后端

### 7.1 方案选择

| 方案 | 结论 | 原因 |
|------|------|------|
| 1. 两套 HiT，谁高置信用谁 | 不选 | 维护成本高，算力翻倍，且结果冲突难统一 |
| 2. RGB 和 Depth 各走完整 HiT，再做 MLP | 不选 | 比方案 1 更合理，但仍然过重 |
| 3. RGB 主干 + 后端深度模块 + 融合头 | 选 | 最符合当前资源预算，也最容易落地 |

### 7.2 最终实现

不实现两套完整 HiT，也不把深度直接压成裸 `depthScore`。  
深度处理、HiT 和 MLP 融合统一封装进 `TrackerBackend`，控制层只消费其输出。

1. `geometry` 同时输出 RGB 和深度对齐视图。
2. RGB 路径进入主 HiT，作为外观主分支。
3. 深度路径进入轻量深度编码器。
   - 可先用 HHA / 三通道深度编码。
   - 可用浅层 CNN / 小型 ViT / 共享前几层 stem。
   - 不复制完整 HiT。
4. `TrackerBackend` 内部进行深度一致性打分与融合。
5. 后端融合头优先用线性权重或小型 MLP。
6. 深度缺失时，后端关闭深度模块，系统自动退化为 RGB-only。

### 7.3 参数来源

| 模块 | 参数来源 | 说明 |
|------|----------|------|
| 深度模块 | `Depth-Anything-V2-Small` 官方 checkpoint | 仅用编码器权重做 warm start；深度图先归一化并转为 3 通道伪图，再进入后端深度模块 |
| 融合 MLP | 随机初始化 | 采用 Xavier / He 初始化 |
| HiT 主干 | 现有 HiT 预训练权重 | 训练阶段保持冻结 |

### 7.4 推荐的融合粒度

| 粒度 | 建议 | 说明 |
|------|------|------|
| 输入级 | 不建议 | 把深度硬拼成三通道图，会丢语义，且和 RGB 分布差异大 |
| 特征级 | 首选 | RGB 主干和深度模块在后端内融合，效果和成本最平衡 |
| 分数级 | 备选 | 适合作为最小可行版本 |
| 双流完整 HiT | 不建议 | 计算和训练成本都过高 |

### 7.5 训练方案

本项目只训练后端中的深度模块和融合 MLP，不回传到 HiT 主干。

1. 输入预处理
   - `frame_depth` 先裁剪到有效深度范围。
   - 再做归一化和缺失值清理。
   - 统一转成 3 通道伪图，与 RGB 保持同一 `ViewSpec`。
2. 冻结策略
   - 冻结全部 HiT 主干参数。
   - 冻结几何层和控制层。
   - 只训练后端深度模块与融合 MLP。
3. 训练数据
   - 优先使用 `rgb_depth` 序列。
   - `rgb_only` 序列只做验证，不更新深度模块。
4. 优化器与学习率
   - `AdamW`
   - 深度模块学习率 `1e-5`
   - 融合 MLP 学习率 `1e-4`
   - `weight_decay = 1e-4`
   - `warmup + cosine decay`
5. 训练顺序
   - 第 1 阶段：只训练融合 MLP 和深度模块末端层。
   - 第 2 阶段：解冻深度模块全部层，继续微调。
   - 第 3 阶段：固定主干，使用验证集早停。
6. 损失组成
   - 跟踪分类损失
   - 边框回归损失
   - IoU 损失
   - 融合一致性损失（可选）
7. 训练结束判定
   - 验证集 AUC / Success 不再上升。
   - 深度模块输出稳定。
   - `rgb_depth` 与 `rgb_only` 都能保持同一接口运行。

### 7.6 完成条件

- HiT 主线只保留一套。
- 深度参与候选筛选和结果融合。
- 模态缺失时系统自动退化。

---

## 8. 问题 6：多帧位置预测接口约束（与第 4 节合并）

本节不再定义另一套独立算法，只补充第 4 节滑动窗口预测模型的接口和验收约束。

### 8.1 方案选择

| 方案 | 结论 | 原因 |
|------|------|------|
| 1. 前 n 帧中心点二维学习 | 不选 | 太弱，处理不了经线、极点和相机运动 |
| 2. 球面坐标 + 深度 + 完整三维建模 | 不选 | 太重，数据需求高 |
| 3. 球面信息 + 深度 + 轻量状态模型 | 选 | 最稳，也最适合现有架构 |

### 8.2 最终实现

`DTC` 维护一个轻量的 `MotionState3D`，状态由最近 `n` 帧共同更新，状态只包含：

- 球面方向
- 距离
- 线速度
- 置信度

更新时使用常速度模型或 Alpha-Beta / Kalman 滤波，不做完整场景三维重建。

### 8.3 预测输入

1. 最近 `n` 帧 `MotionState3D`
2. 最近 `n` 帧球面方向
3. 最近 `n` 帧目标深度摘要
4. 最近 `n` 帧 BFoV 内深度摘要
5. 最近 `n` 帧置信度
6. 最近 `n` 帧 360VOT 球面方向变化率
7. 可选相机运动历史

### 8.4 预测输出

每帧输出：

- `predictedCenter`
- `predictedDepth`
- `predictedVelocity`
- `predictedFov`
- `searchRadius`

### 8.5 计算原则

1. 先基于前 `n` 帧窗口拟合趋势，不允许只用单帧决策。
2. 再把窗口内球面方向转成单位向量序列。
3. 再把窗口内深度转成相对距离序列。
4. 用常速度模型或 Alpha-Beta / Kalman 在窗口上预测下一帧中心。
5. 用窗口内角速度和深度跳变共同决定搜索视场。
6. 深度缺失时仍必须使用前 `n` 帧方向序列完成退化预测。

### 8.6 完成条件

- `DTC` 能稳定输出下一帧预测中心。
- 搜索窗口能随深度和运动变化自动调整。
- 无深度时仍能正常预测。

---

## 9. 执行顺序

### 第 1 步
先改 `docs/interface.md`，把深度字段、运动状态和预测字段补齐。

### 第 2 步
再改 `docs/data.md` 和 `src/instatarget/data/`，把双模式数据源固定下来。

### 第 3 步
再改 `docs/modules/geometry.md` 和 `src/instatarget/geometry/`，保证 RGB 与深度同步裁剪。

### 第 4 步
再改 `docs/modules/controller.md` 和 `src/instatarget/controller/`，把 DTC 的预测算法写实。

### 第 5 步
再改 `docs/modules/tracker.md` 和 `src/instatarget/tracker/`，让后端吃到成对视图。

### 第 6 步
再改 `docs/modules/controller.md`、`docs/modules/tracker.md`，把后端融合与轻量预测落成接口。

### 第 7 步
最后改 `src/instatarget/app/`、`process.md` 和相关测试，接通完整流程。

---

## 10. 验收标准

- `geometry` 不丢深度。
- `DTC` 能用 BFoV 深度和前后帧深度变化预测下一帧位置。
- `TrackerBackend` 输入接口保留深度字段，内部完成深度处理、HiT 和融合。
- `rgb_depth` 和 `rgb_only` 都能跑通。
- 第 3 点采用单主 HiT + 后端深度模块 + 融合头，不实现两套完整 tracker。
- 第 6 点采用球面方向 + 深度 + 轻量状态模型，不做完整三维重建。
- 文档、接口、流程三者一致。
