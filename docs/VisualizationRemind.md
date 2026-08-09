# Visualization 后续阶段修改提醒

> 本文档记录后续功能落地时，可视化模块需要接入或修改的位置。基本原则是：优先在应用编排层
> 增加调用，只有核心数据契约或输出类型发生变化时才修改 `instatarget.visualization`。

---

## 1. 深度转颜色模块（已落地）

如果深度模块输出以下格式，则 **不需要修改 visualization 代码**：

```python
Mapping[int, NDArray[np.uint8]]  # viewId -> RGB [H, W, 3]
```

深度转颜色由 `TrackerBackend` 内部完成；应用编排层若要记录结果，只需把后端已经生成的
`viewId -> RGB` 映射传给 visualization：

```python
depthRgbByViewId = diagnosticDepthRgbByViewId  # 由后端诊断钩子导出，具体接口另行定义
recorder.recordDepthRgb(frame, depthRgbByViewId)
```

`recordDepthRgb()` 会原样保存结果，不进行归一化、颜色映射或通道转换。

只有出现以下变化时才需要修改
`src/instatarget/visualization/recorder.py::recordDepthRgb()`：

| 变化 | 需要的修改 |
|---|---|
| 深度模块返回 `Sequence`，而不是按 `viewId` 建立的 `Mapping` | 在应用层建立 `viewId -> depthRgb` 映射；优先不改 visualization |
| 输出是 BGR | 在深度模块或应用边界转换为 RGB；禁止让 visualization 猜测颜色顺序 |
| 输出是浮点张量或 GPU tensor | 在深度模块边界转换为 CPU `uint8 RGB`；visualization 保持只负责写盘 |
| 输出不再按局部视图区分 | 先定义稳定的图像 ID，再扩展文件命名和记录器参数 |
| 需要同时保存转换前深度数据 | 新增独立 stage，不要改变现有 `depth_rgb` 的含义 |

---

## 2. Runtime driver 落地

当前 `src/instatarget/app/driver.py` 仍是占位文件。完整运行流水线实现后，需要在 `app` 层加入
以下四个采集点：

| 计算完成时机 | 调用 |
|---|---|
| geometry 生成 `LocalView` 后 | `recordLocalRgb(frame, views)` |
| `TrackerBackend` 完成深度 RGB 转换后 | `recordDepthRgb(frame, depthRgbByViewId)` |
| backend 生成 `LocalObservation` 后 | `recordBackendBoxes(frame, views, localObservations)` |
| local box 完成 geometry 回投影后 | `recordGeometryBoxes(frame, projectedObservations)` |

记录器应由应用层根据 `config.visualization` 创建一次并复用。不要把记录调用放入 geometry、tracker
或 controller，以免这些模块反向依赖调试功能。

---

## 3. 深度后端和融合头（已落地）

双流 HiT 或 MLP 融合已经由 `TrackerBackend` 承担。如果 `LocalObservation.bbox` 仍代表后端最终局部框，则
`backend_box` 不需要修改，会自动显示融合后的最终候选。

如果需要同时比较 RGB 分支框、深度分支框和融合框，则需要：

1. 在核心类型中先定义三个框的明确契约，禁止依赖无类型字典。
2. 在配置的 `visualization.stages` 中增加独立阶段名。
3. 更新 `VISUALIZATION_STAGES`。
4. 在 `VisualizationRecorder` 中增加对应记录方法和独立目录。
5. 增加每个新阶段的像素与路径测试，并同步更新 `docs/visualization.md`。

不要复用 `backend_box` 文件覆盖多个不同语义的框，否则无法进行人工对照。

---

## 4. Controller 与 geometry 契约变化

出现以下变化时需要检查 visualization：

| 契约变化 | 检查位置 |
|---|---|
| `LocalObservation.bbox` 不再是局部视图 XYWH 像素坐标 | `recordBackendBoxes()` 与 `drawBoxRgb()` |
| `ProjectedObservation.bbox` 不再是 ERP XYWH 像素坐标 | `recordGeometryBoxes()` |
| ERP 水平坐标不再允许跨经线 | `drawBoxRgb(..., wrapHorizontal=True)` |
| 一个 view 产生多个候选框 | 文件命名需加入候选 ID，并定义是否合并到同一张图 |
| DTC 每帧产生 guard/adaptive/recovery 多视图 | 应用层按 `viewId` 记录，不改变四个既有 stage 的语义 |
| 最终框改由 `TrackResult` 独立产生 | 新增最终结果 stage，不要把它混同为 `geometry_box` |

坐标契约变化时，必须先修改核心类型和 geometry 测试，再调整可视化；不能在绘图函数中静默猜测
坐标格式。

---

## 5. 并发与性能阶段

完整并发流水线启用后，当前 PNG 原子替换可以避免产生半写文件，但调用线程仍会承担 PNG 编码和
磁盘 I/O。若可视化明显影响跟踪吞吐量，再增加有界写入队列和单独写线程，并遵守以下约束：

1. 队列容量进入 runtime 或 visualization 配置并严格校验。
2. 每个任务携带 sequence、frame、stage 和 view ID。
3. 诊断模式可以明确选择背压或丢弃；比赛模式保持 visualization 关闭。
4. `finalize()` 必须等待已接受的图片全部写完，并传播写入异常。

在性能数据证明同步写入成为瓶颈之前，不需要提前修改当前实现。

---

## 6. 修改检查表

后续阶段每次修改数据流时检查：

- 输入是否仍是 CPU `uint8 RGB [H, W, 3]`。
- `viewId` 是否仍能唯一关联视图、后端观测和 geometry 结果。
- 局部框与 ERP 框的坐标空间是否仍与现有方法一致。
- 是否只需在 app 增加调用，而不需要修改 visualization。
- 新增 stage 后是否同步更新配置校验、两份 YAML、测试和 `docs/visualization.md`。
- 关闭 `visualization.enabled` 后是否仍然不创建目录、不复制数组、不改变计算结果。
