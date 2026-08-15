# Visualization 模块结构

Visualization 只读取帧、视图、观测和结果生成诊断产物，不参与算法决策。

| 文件 | 职责 |
|---|---|
| `recorder.py` | 四阶段中间产物 |
| `result.py` | 最终 ERP 结果图 |
| `image.py` | 框、标签和跨缝绘制 |
| `instance_ids.py` | 实例分组与清单 |
| `time_counter.py` | 处理区间时间产物 |

深入阅读：[stageArtifacts.md](stageArtifacts.md)、[resultRendering.md](resultRendering.md)、[instanceIds.md](instanceIds.md)、[processingTiming.md](processingTiming.md)、[parameters.md](parameters.md)。
