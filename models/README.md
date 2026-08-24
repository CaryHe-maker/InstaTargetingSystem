# ARTrackV2-B-256 模型产物

将官方 `ARTrackV2-B-256` checkpoint 通过 Git LFS 获取到本目录，文件名固定为
`artrackv2_b_256.pth.tar`。文件应包含 `net` state dictionary；代码会严格加载 ViT-B、
256 搜索尺寸对应的权重。

当前发布文件 SHA-256：
`a99b7f8086e4827ecfe32ec8a9d32ad41c1ca9ff3cac551b62ec95576ca01d05`

ARTrackV2 的分数不是校准产物，首次部署前可使用项目的验证集重新拟合
`ScoreCalibration`，再把生成的 JSON 写入 `configs/RGBonly.yaml` 的
`scoring.calibrationArtifact`。没有校准文件时，开发调用可显式使用
`buildRuntime(..., allowUncalibratedScoring=True)`。
