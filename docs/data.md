# InstaTargetingSystem 数据规范

> 本文档定义 AirSim360 数据入口、样本组织、伪标注生成和数据与接口的对应关系。  
> 数据层只负责读取、对齐和派生，不参与跟踪决策。

---

## 1. 数据范围

| 数据源 | 内容 | 用途 |
|------|------|------|
| AirSim360 | ERP RGB、Depth、semantic mask、instance mask | 主训练、验证、回归测试 |
| 真实全景数据 | ERP RGB 与官方初始框 | 泛化测试 |
| 360VOT | 全景标注与评测样例 | 几何回归和属性分析 |
| 官方比赛数据 | 仅在评测期使用 | 最终提交 |

AirSim360 是当前主数据源。它提供多模态全景帧，但不直接提供比赛格式的逐帧跟踪框，
因此需要由本项目的数据层生成临时目标序列。

---

## 2. 帧级数据包

AirSim360 的同帧数据应按同一 `frameIndex` 对齐读取：

| 字段 | 内容 | 约束 |
|------|------|------|
| `frame_rgb_erp` | ERP 彩色全景图 | 必需，RGB `uint8` |
| `frame_depth` | 深度图 | 可缺省，使用有效掩码 |
| `frame_semantic_mask` | 语义类别图 | 可缺省 |
| `frame_instance_mask` | 实例 ID 图 | 可缺省，但训练推荐保留 |
| `semantic_class_list` | 类别表 | 用于解释类别 ID |

同一像素在四个图层中表示同一球面方向。若某模态缺失，必须显式标记，不允许填充伪造值。

---

## 3. 数据模式

| 模式 | 输入 | 行为 |
|------|------|------|
| `rgb_only` | 只有颜色图 | RGB HiT + 球面几何；深度权重退化为 0 |
| `rgb_depth` | 颜色图 + 深度图 | RGB HiT + 深度预处理/分支 + 后端融合头；DTC 只消费摘要和分数 |

---

## 4. 由数据派生的样本

### 4.1 初始框

当 `instance_mask` 可用时，首帧目标框由目标实例的外接矩形生成。

### 4.2 伪真值

当目标实例跨帧稳定时，可为每帧生成伪真值：

- 目标实例存在时，输出 `BBoxXYWH` 和 `visible=true`。
- 目标实例不存在时，输出空框或 `visible=false`。
- 若实例 ID 不稳定，按外观和位置连续性重建临时 `trackId`。

### 4.3 深度摘要

`frame_depth` 由数据层以 `DepthPlane(values, validMask, unit)` 提供；数据层不做颜色化或跟踪决策。
局部深度摘要由 `TrackerBackend` 的 `DepthProcessor` 根据局部预测框生成，并通过
`LocalObservation.depthSummary` 返回给 DTC。摘要至少包含中位数、均值、有效率、最小/最大深度
和置信度。有效率不足、值非有限或深度跳变异常时返回 `None` 或低置信摘要，DTC 不更新 range 状态。

RGB-only 序列必须令 `FramePacket.depth=None`，而不是创建形状相同的零深度图；后端和 DTC
应自动退化为 RGB-only + 球面运动预测。

---

## 5. 推荐目录

```text
data/
  AirSim360/
    sequence_id/
      rgb/
      depth/
      semantic/
      instance/
      meta.json
  real360/
  360vot/
```

`meta.json` 可记录序列名、模态是否存在、单位和类别表路径。

---

## 6. 接口对齐

本数据文档对应 `interface.md` 中的：

- `AirSim360DataSource`
- `AirSim360Record`
- `FramePacket`
- `PseudoTrackBuilder`

数据层只负责 `open/read/close` 和样本派生，不负责 `DTC`、`HiT` 或几何回投影。
