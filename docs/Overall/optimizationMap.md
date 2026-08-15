# 优化入口地图

## 想提高找回率

先检查 Controller 的视域覆盖和 StateEvaluator，而不是直接放大 HiT 框。相关文档是 `Controller/viewPlanning.md`、`Controller/stateEvaluator.md` 和 `Controller/stateMachine.md`。主要参数是 `successRate`、两个 overlap 阈值、单帧视图预算和 UNCERTAIN 耐心。

## 想降低误检

检查三层分数：Tracker 原始模型分数、Beta Calibration 后的局部分数、双框融合分数。不要只提高最终阈值，因为这可能增加 Round 3 和 LOST 次数。应同时观察可靠融合比例、单框输出比例、状态停留时间和校准曲线。

## 想降低延迟

主要成本通常是视图数量乘以 HiT 推理时间。先统计每种状态的平均轮数，再考虑模型精度、视图尺寸或批处理。可视化与 sink 已排除在 `time.json` 外，因此不能通过关闭可视化伪造算法延迟改善。

## 想改善快速运动

查看 `Controller/motionPredictor.md`。优先验证时间戳、可靠样本是否进入历史、切平面残差和速度裁剪；不要让预测输出反向写入测量历史。

## 想改善球面边界问题

查看 `Geometry/seamHandling.md` 和 `Geometry/coordinateTransforms.md`。任何 ERP bbox 的交集、IoU 或显示逻辑都必须使用循环横坐标，不能直接套普通平面区间。

## 优化的最低验证集

每次算法调整至少比较：成功率/IoU、平均每帧视图数、每状态帧数、可靠融合比例、平均处理时间和最差处理时间。涉及分数时应保留原始分数与校准分数的联合日志，避免只看最终输出。

