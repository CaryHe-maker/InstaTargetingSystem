# 快速使用

1. 安装 Git LFS 并执行 `git lfs pull`，确认 `models/artrackv2_b_256.pth.tar` 为完整实体文件。
2. 安装 `requirements.txt` 中与 PostTrainV2.4 相同的依赖。
3. 使用 `configs/RGBonly.yaml` 运行 AirSim360 数据；默认数据根目录可指向
   `E:\NewDownload\train`。

首次获得验证结果后，使用 `tools/collect_score_calibration.py` 和
`tools/fit_score_calibration.py` 生成 ARTrackV2 专用校准文件，再写回配置。
