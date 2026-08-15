# Overall 模块结构

系统把 ERP 全景帧转换为局部透视视图，使用 HiT 产生候选，再由 Controller 在球面空间完成融合、状态转移和结果提交。

## 模块关系

| 模块 | 主要职责 |
|---|---|
| Core | 公共类型、协议、配置和异常 |
| Runtime | 组件装配、逐帧线程和资源生命周期 |
| Controller | 运动预测、视域规划、候选评估和状态机 |
| Tracker | HiT 推理、深度处理、RGB-D 融合和模板缓存 |
| Geometry | ERP、球面、BFoV 和局部透视坐标转换 |
| Data | 帧源、AirSim360 解析、伪真值和结果落盘 |
| Visualization | 中间视图、最终结果、实例 ID 和计时产物 |
| Evaluation | 平面/球面指标与性能统计 |
| Competition | InstaTest 输入发现、运行与 BFoV 输出 |
| Training | 训练样本契约和当前训练边界 |

## 深入阅读

- [architecture.md](architecture.md)：端到端数据流和依赖方向。
- [runtimeThread.md](runtimeThread.md)：从 CLI 到每帧提交的完整运行线程。
- [optimizationMap.md](optimizationMap.md)：常见优化目标应该修改哪里、观察什么指标。

公共入口为 `track.py`、`src/instatarget/app/commands.py` 和 `src/instatarget/app/driver.py`。

