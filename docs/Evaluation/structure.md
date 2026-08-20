# Evaluation 模块结构

Evaluation 消费预测和真值，计算平面、循环 ERP、球面和性能指标，不修改运行时状态。

| 文件 | 职责 |
|---|---|
| `eval/otb_metrics.py` | bbox/循环 IoU、成功曲线、AUC 和 tracking loss rate |
| `eval/spherical_metrics.py` | 球面中心误差和 BFoV IoU |
| `eval/profiler.py` | 命名代码段时间统计 |
| `tools/run_airsim360_dataset.py` | 数据集运行、真值和产物汇总 |
| `tools/eval_manifest_controller.py` | `E:\NewDownload\train` 单序列生产 Controller 评估 |
| `tools/summarize_manifest_evaluations.py` | 多序列报告连续性检查与聚合 |

深入阅读：[trackingMetrics.md](trackingMetrics.md)、[profiling.md](profiling.md)、[verificationWorkflow.md](verificationWorkflow.md)。

