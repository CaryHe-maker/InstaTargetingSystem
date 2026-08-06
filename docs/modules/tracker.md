# InstaTargetingSystem 跟踪后端规范

> 本文档定义 HiT 跟踪后端的输入、输出、模板命令和实现边界。
> 后端只做局部目标匹配，不做全景几何和状态机决策。

---

## 1. 职责边界

| 输入 | 输出 | 不负责 |
|------|------|--------|
| `LocalView`、`TemplateCommand` | `LocalObservation` 序列 | 球面状态、深度门控、全局恢复 |

后端以 HiT 为主，也可以替换成 HiT-Tiny、DyHiT、ONNX Runtime 或 TensorRT 版本。
它们都必须满足同一 `TrackerBackend` 契约。

---

## 2. 后端生命周期

### 2.1 `initialize()`

初始化模板特征。每个序列只调用一次。输入为模板视图和模板框。

### 2.2 `infer()`

对一个或多个局部视图执行推理，返回同序的局部观测。输出至少包含：

- 局部框
- 模型分数
- 外观分数
- 推理耗时

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

1. 只接收局部透视 RGB 图。
2. 不接收完整 ERP 图作为主输入。
3. 若输入携带深度，后端必须忽略它，深度只供控制层使用。
4. 视图顺序必须和 `views` 顺序一致。

---

## 5. 输出观测规则

`LocalObservation` 应包含：

- `viewId`
- `bbox`
- `modelScore`
- `appearanceScore`
- `latencyNs`

后端不得输出 BFoV、全局状态或恢复策略。回投影和融合由控制层完成。

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
