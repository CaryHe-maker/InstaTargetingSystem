# Runtime 模块结构

Runtime 是组合根和顺序执行器，本身不实现状态判定或模型算法。

| 文件 | 职责 |
|---|---|
| `src/instatarget/app/driver.py` | 组件装配与逐帧多轮循环 |
| `src/instatarget/app/commands.py` | 用户命令和路径解析 |
| `src/instatarget/app/track.py` | 通用序列入口 |
| `src/instatarget/app/track_airsim360.py` | AirSim360 生命周期与时间产物 |

深入阅读：[runtimeWiring.md](runtimeWiring.md)、[trackingLoop.md](trackingLoop.md)、[lifecycleAndTiming.md](lifecycleAndTiming.md)、[parameters.md](parameters.md)。
