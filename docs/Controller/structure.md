# Controller 模块结构

Controller 负责“看哪里、相信哪个候选、是否继续查询、下一帧处于什么状态”。

| 文件 | 职责 |
|---|---|
| `depth_aware_track_controller.py` | 状态所有权、帧事务与原子提交 |
| `state_model.py` | 状态、证据、候选、事务数据结构 |
| `state_machine.py` | 跨帧纯状态转移 |
| `state_evaluator.py` | 每轮候选池评估、Fusor 调用和测量接受资格 |
| `fusor.py` | seam-aware 两框融合和单一最佳候选选择 |
| `classifier.py` | 保留的 30° 球面聚类工具；当前生产路径不调用 |
| `recovery_planner.py` | 四角视图与 cubemap 规划 |
| `motion_estimator.py` | 球面多帧运动预测 |
| `fused_score.py` | 外观/运动校准与 70/30 SingleScore 合成 |
| `template_policy.py` | 固定第 0 帧 anchor 的 KEEP 策略 |

深入阅读：[stateMachine.md](stateMachine.md)、[motionPredictor.md](motionPredictor.md)、[viewPlanning.md](viewPlanning.md)、[stateEvaluator.md](stateEvaluator.md)、[scoreCalibration.md](scoreCalibration.md)、[templateAndTransaction.md](templateAndTransaction.md)、[parameters.md](parameters.md)。
