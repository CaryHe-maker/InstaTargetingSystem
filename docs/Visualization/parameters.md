# Visualization 参数索引

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `visualization.enabled` | false | 是否创建中间阶段产物 |
| `visualization.outputRoot` | `../outputs/visualization` | 输出根，相对 YAML 路径解析 |
| `visualization.stages` | 三阶段全集 | 选择 local_rgb、backend_box、geometry_box |

最终结果图、InstanceID 文本和工具输出目录由各命令参数决定。绘图颜色、线宽、标签间距位于 `visualization/image.py` 的代码常量，只影响显示，不属于跟踪超参数。

启用任何阶段都不会改变 `time.json.elapsed*`，因为写图发生在处理区间外；但会增加进程总墙钟和磁盘用量。

