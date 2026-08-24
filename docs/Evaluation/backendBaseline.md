# Backend Baseline

现有基线结果目录（本机实际路径）为
`E:\tringData\shared_control\production2`；用户消息中的
`E:\tringData\shared\_control\production2` 在当前机器上不存在，已按实际目录读取。

该目录包含 10 个 `report.json`，对应 `E:\NewDownload\train` 的 validation 序列。
按序列简单平均的基线为：

| 指标 | 基线 |
|---|---:|
| circular ERP mean IoU | 0.25618 |
| spherical mean IoU | 0.22360 |
| success AUC | 0.25640 |

后续 ARTrackV2 评估必须使用相同的序列集合、可见帧筛选和 report schema，
并同时记录每序列值与全局/按帧加权值；不能只比较单个序列的峰值。
