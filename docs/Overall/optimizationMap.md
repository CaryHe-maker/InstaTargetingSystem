# 优化入口地图

## 想提高找回率

先检查 Controller 的视域覆盖和 StateEvaluator，而不是直接放大 HiT 框。相关文档是 `Controller/viewPlanning.md`、`Controller/stateEvaluator.md` 和 `Controller/stateMachine.md`。当前有效门限是 `tracking.candidateMinScore` 与 `evaluator.fusionSourceMinConfidence`；Fusor overlap 固定为代码常量 0.70。视图预算为 TRACKING 8、UNCERTAIN 10、LOST 12，当前没有单独的 UNCERTAIN patience 参数。

## 想降低误检

检查四层分数：`backendFusedScore`、`appearanceProbability`、有效运动概率、`singleScore`，再检查双框融合分数。不要只提高最终门限，因为它会降低测量接受率并改变后续状态分布。当前同帧最多两轮，不存在 Round 3。应同时观察候选排序 AUC、Brier score、可靠融合比例、单框输出比例、状态停留时间和校准曲线。

## 想降低延迟

HiT 已按轮执行真实 tensor batch：TRACKING 为 4、4，UNCERTAIN 为 6、4，LOST 为 12；RGB-D 每轮还有一个 depth batch。批处理减少模型 forward 次数，但预处理、显存占用和部分模型算术仍随图片数增长。应先统计每种状态的轮数、batch size、forward 数、GPU 利用率与峰值显存，再评估精度、视图尺寸和更深层并行。可视化与 sink 已排除在 `time.json` 外，因此关闭可视化不会改变算法计时。

## 想改善快速运动

查看 `Controller/motionPredictor.md` 和 `motionProjectionUpgrade.md`。优先验证时间戳、可靠样本、中心/尺度协方差、reliability、切平面残差和速度裁剪；不要让预测输出反向写入测量历史。

## 想改善球面边界问题

查看 `Geometry/seamHandling.md` 和 `Geometry/coordinateTransforms.md`。先比较直接 bbox、间接 bbox 和 `envelopeInflation`；任何 ERP bbox 的交集、IoU 或显示逻辑都必须使用循环横坐标，不能直接套普通平面区间。

## 优化的最低验证集

每次算法调整至少比较：成功率/IoU、平均每帧视图数、每状态帧数、可靠融合比例、平均处理时间和最差处理时间。涉及分数时应保留原始分数与校准分数的联合日志，避免只看最终输出。

