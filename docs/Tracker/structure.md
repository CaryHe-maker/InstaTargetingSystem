# ARTrackV2 Tracker

运行时由三层组成：`PyTorchARTrackV2Session` 负责官方 ARTrackV2-B-256 网络，
`ARTrackBackend` 做批量输入/输出校验，`TrackerBackendImpl` 将模板 revision、
局部框和 controller 的视图事务串起来。

ARTrackV2 使用 128x128 模板、256x256 搜索 crop 和 400-bin 自回归坐标。每个
perspective view 都在自己的局部坐标系中以模板框为中心生成搜索 crop，输出再映射
回 `LocalView`，随后由现有球面 Geometry 和 Controller 完成 IoU 优化相关的融合。

官方代码位于 `src/instatarget/vendor/artrackv2`，运行时依赖保持项目已有的
PyTorch、torchvision、timm、easydict 和 OpenCV 版本。
