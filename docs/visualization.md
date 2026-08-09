# Visualization 中间结果可视化

> 本模块用于人工检查跟踪过程和优化超参数。它只把已有计算结果写成 PNG，不参与推理、
> 深度颜色化、geometry 转换、候选排序或最终决策。

---

## 1. 可视化内容

| stage | 输入 | 输出图 |
|---|---|---|
| `local_rgb` | `LocalView.rgb` | 每一步 geometry 生成的局部 RGB 视图 |
| `depth_rgb` | `TrackerBackend` 深度预处理生成的 RGB 数组 | 已转换深度 RGB 图的无损副本 |
| `backend_box` | `LocalView.rgb + LocalObservation.bbox` | 后端局部目标框 |
| `geometry_box` | `FramePacket.rgb + ProjectedObservation.bbox` | geometry 回投影后位于 ERP 原图上的目标框 |
| `dtc_candidates`（计划新增） | DTC 的候选簇和 `decisionScore` | 单帧候选聚合诊断 |
| `dtc_state`（计划新增） | `TrackResult.status`、门控分数和视图角色 | 状态机与视图预算诊断 |
| `dtc_prediction`（计划新增） | `predictedMotion`、预测 BFoV 和恢复半径 | 多帧预测诊断 |

所有目标框固定使用荧光绿 `#39FF14`。`geometry_box` 支持跨 ERP 经线的框，框会同时出现在
原图左右边界。原始数组不会被原地修改。

`dtc_*` 是第五阶段的可选旁路，不参与候选排序、状态提交或结果输出；启用前必须同步扩展
`VISUALIZATION_STAGES`、配置 schema、记录器方法和路径测试。当前实现默认只承诺前四个 stage。

深度 RGB 的生成不属于本模块。它由 `TrackerBackend` 的深度链路完成；`recordDepthRgb()` 只接受并原样保存已有模块输出的
`uint8 [H, W, 3]` RGB 数组，不执行归一化、伪彩色映射或其他深度处理。

---

## 2. 开启与选择

两份运行配置均包含以下段落，默认关闭：

```yaml
visualization:
  enabled: false
  outputRoot: ../outputs/visualization
  stages:
    - local_rgb
    - depth_rgb
    - backend_box
    - geometry_box
```

将 `enabled` 改为 `true` 即可记录中间结果。只保留需要的 `stages` 可以减少 PNG 编码和磁盘
I/O，例如只检查 geometry 回投影：

```yaml
visualization:
  enabled: true
  outputRoot: ../outputs/visualization
  stages:
    - geometry_box
```

`outputRoot` 的相对路径以 YAML 文件所在目录为基准解析。关闭时所有记录方法立即返回空元组，
不会创建输出目录，也不会复制图像。

---

## 3. 调用方式

从应用编排层创建一个记录器，在对应计算完成后调用阶段方法：

```python
from instatarget.visualization import VisualizationRecorder

recorder = VisualizationRecorder(config.visualization)

views = geometry.cropViews(frame, plan.views)
recorder.recordLocalRgb(frame, views)

# depthRgbByViewId 直接来自现有深度转 RGB 模块，不由 visualization 生成。
recorder.recordDepthRgb(frame, depthRgbByViewId)

localObservations = backend.infer(views, plan.templateCommand)
recorder.recordBackendBoxes(frame, views, localObservations)

# projectedObservations 是应用层完成 local box -> BFoV -> ERP bbox 后的结果。
recorder.recordGeometryBoxes(frame, projectedObservations)
```

四个方法均返回实际写出的 `tuple[Path, ...]`，便于测试或诊断工具直接索引产物。记录器按
`viewId` 关联局部视图和后端观测；重复 ID 或找不到对应局部视图时会立即报协议错误。

---

## 4. 输出组织

输出固定为无损 RGB PNG。PNG 能逐像素保留荧光绿框和已有深度颜色化结果，也避免 JPEG 压缩
干扰人工判断。目录结构如下：

```text
<outputRoot>/
  <sequenceId>/
    frame_000007/
      local_rgb/
        view_0003.png
      depth_rgb/
        view_0003.png
      backend_box/
        view_0003.png
      geometry_box/
        view_0003.png
```

`sequenceId` 中不适合文件名的字符会替换为下划线。相同 sequence、frame、stage 和 view 再次
写入时会原子替换旧文件，因此一次运行中每个路径始终代表该步骤的最新完整结果。

---

## 5. 模块组织

```text
src/instatarget/visualization/
  __init__.py    # 公共导出
  recorder.py    # 阶段选择、数据关联、路径组织
  image.py       # 荧光绿框绘制和 ERP 水平循环处理
  png.py         # 无损 PNG 原子写入
```

依赖方向为 `core <- visualization <- app`。可视化模块只读取核心数据类型，不被 geometry、
tracker 或 controller 反向依赖。这样关闭模块时原有计算链路保持不变，后续完整 runtime driver
落地后也只需在应用编排层增加上述调用点。

---

## 6. 使用约束

1. `depthRgbByViewId` 必须是现有深度转换结果，颜色顺序为 RGB，不是 BGR。
2. 所有输入图像必须为非空 `uint8 [H, W, 3]` CPU NumPy 数组。
3. `LocalObservation.bbox` 使用局部视图坐标；`ProjectedObservation.bbox` 使用 ERP 原图坐标。
4. 本模块面向诊断运行，不建议在正式计时或比赛输出模式中开启。
5. `outputs/` 已由仓库忽略规则排除，生成图片不会进入版本控制。
