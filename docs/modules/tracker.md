# InstaTargetingSystem 跟踪后端规范

> 本文档定义 HiT 跟踪后端的输入、输出、模板命令和实现边界。  
> 当前后端同时支持 RGB-only 和可选 RGB-D 形态；深度伪彩色、第二分支和融合头均在 backend 内部完成，控制层只消费统一观测契约。

---

## 1. 职责边界

| 输入 | 输出 | 不负责 |
|------|------|--------|
| `LocalView`、`TemplateCommand` | `LocalObservation` 序列 | 球面状态、全局恢复和搜索决策 |

后端以 HiT 为主，也可以替换成 HiT-Tiny、DyHiT、ONNX Runtime 或 TensorRT 版本。  
深度链路是可选依赖；未配置深度组件时严格退化为 RGB-only 契约。

---

## 2. 后端生命周期

### 2.1 `initialize()`

初始化模板特征。每个序列只调用一次。输入为模板视图和模板框。

### 2.2 `infer()`

对一个或多个局部视图执行推理，返回同序的局部观测。RGB-only 模式的流程为：

1. 读取局部 RGB。
2. 校验模板命令与模板版本。
3. 将 RGB 送入 HiT 主干做局部匹配。
4. 输出裁剪后的局部框与外观分数。
5. 将 `depthScore` 固定为 `0.0`，`fusedScore` 固定为 `appearanceScore`，`depthSummary` 固定为 `None`。

配置深度组件时，后端额外对齐并归一化深度、生成单调浮雕伪彩色图，调用第二分支并由融合头统一产生 `depthScore` 与 `fusedScore`；深度不可用的单个视图仍按 RGB-only 退化。

输出至少包含：

- 局部框
- 模型分数
- 外观分数
- 深度分数
- 融合分数
- 深度摘要
- 推理耗时

第四阶段预留的深度分支流程是：

1. 读取局部 `depth`。
2. 进行对齐、归一化、缺失值掩码与浮雕式伪彩色编码。
3. 再把深度伪彩色图送入第二个 HiT。
4. 以 MLP 融合 RGB HiT、深度 HiT、模板上下文和轻量几何参数。

### 2.3 `close()`

释放设备资源和缓存。序列结束前必须调用。

---

## 3. 模板命令

| 命令 | 含义 |
|------|------|
| `KEEP` | 不更新模板 |
| `UPDATE_RECENT` | 更新近期模板 |
| `UPDATE_STABLE` | 更新稳定模板 |
| `RESET_TO_ANCHOR` | 清空动态模板，只保留首帧模板 |

模板命令只由 DTC 决定。后端只负责执行，不负责判断何时更新。

---

## 4. 输入视图规则

1. 只接收局部透视视图，不接收完整 ERP 图作为主输入。
2. 视图对象可以携带与 RGB 对齐的 `depth` 字段；只有启用深度组件时才消费该字段。
3. 视图顺序必须和 `views` 顺序一致。

---

## 5. 输出观测规则

`LocalObservation` 应包含：

- `viewId`
- `bbox`
- `modelScore`
- `appearanceScore`
- `depthScore`
- `fusedScore`
- `depthSummary`
- `latencyNs`

后端不得输出 BFoV、全局状态或恢复策略。回投影和搜索规划由控制层完成。

RGB-only 模式的 `LocalObservation` 额外约束是：

- `depthScore == 0.0`
- `fusedScore == appearanceScore`
- `depthSummary is None`

---

## 6. 模型形态

| 形态 | 作用 |
|------|------|
| `HiT-Small` | 主力实时后端 |
| `HiT-Tiny` | 更轻量的速度版本 |
| `DyHiT` | 复杂帧加速版本 |
| `ONNX Runtime` | 部署后端 |
| `TensorRT` | 高性能部署后端 |

这些形态可以共享同一个逻辑接口，只是实现后端不同。
当前仓库已落地 RGB-only 和 RGB-D 两种形态；深度分支仍可由配置关闭。

---

## 7. 后端内部算法

### 7.1 当前实现

- 默认读取局部 RGB；启用深度组件时同时读取对齐的局部深度。
- 使用 HiT 主干输出局部框与外观分数，并可调用第二分支。
- 观测层统一写成 `LocalObservation`。

### 7.2 RGB 主干

- RGB 局部图进入 HiT 主干。
- 主干输出局部框候选和外观特征。

### 7.3 融合头

- RGB-only 时不启用深度融合，`fusedScore` 退化为 `appearanceScore`。
- RGB-D 时把 RGB 分数、深度分数和模板上下文送入轻量融合头。

### 7.4 训练约束

- 当前实现不在推理后端内训练深度分支或融合头。
- 深度分支先做浮雕式伪彩色编码，融合头可在独立训练链路中替换。
- 控制层不参与深度编码和 MLP 训练。

### 7.5 深度分支实现顺序

该深度颜色化模块仍放在 `TrackerBackend` 内部，由后端统一完成，不上移到控制层。

启用深度分支时采用下面的顺序：

1. 深度图先做中值 / 分位数归一化，再估计局部背景面。
2. 计算“浮雕量”，让更近的站立目标在图上更亮、更凸。
3. 叠加边缘增强，让轮廓更硬、更容易被 HiT 分辨。
4. 用单调伪彩色生成深度 RGB 图，近亮远暗，不使用彩虹色图。
5. 深度 RGB 图再进入第二个 HiT。
6. RGB HiT 输出、深度 HiT 输出、模板上下文和轻量几何参数一起进入 MLP。
7. 融合头单独训练，初始值建议偏向 RGB，例如 RGB=0.70、Depth=0.20、Context=0.10。

---

## 7. 失败处理

- 模板未初始化时不得推理。
- 观测为空时返回空序列，不得补造框。
- 推理错误必须上抛，不得吞掉。
- 设备异常时由上层线程转换为 `FatalError`。

---

## 8. 接口对齐

本跟踪文档对应 `interface.md` 中的：

- `TrackerBackend`
- `LocalView`
- `LocalObservation`
- `TemplateCommand`
- `TemplateCommandKind`

后端是 `T2` 的唯一职责，不参与 `DTC` 的状态决策。
