# ARTrackV2-B-256 模型产物

将官方 `ARTrackV2-B-256` checkpoint 下载为 `artrackv2_b_256.pth.tar` 放在本目录。
文件应包含官方 checkpoint 的 `net` state dictionary；代码会严格加载 ViT-B、256 搜索尺寸对应的权重。

ARTrackV2 的分数不是旧模型的 calibrated artifact，首次部署前应使用项目的验证集重新拟合
`ScoreCalibration`，再把生成的 JSON 写入 `configs/RGBonly.yaml` 的
`scoring.calibrationArtifact`。没有校准文件时，开发调用可显式使用
`buildRuntime(..., allowUncalibratedScoring=True)`。
