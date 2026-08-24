# ARbackendV1 发布报告

## 提交范围

- 基线：`main` / `PostTrainV2.4`，提交 `65d7da4`。
- 目标分支：`NewBackendV1`。
- 后端：官方 `ARTrackV2-B-256`，作为 ARbackendV1 的唯一推理后端。
- 依赖：保持 PostTrainV2.4 的 PyTorch、torchvision、timm、NumPy、OpenCV、PyYAML 和 easydict 版本约束。
- Controller、球面 Geometry、I/O 和结果协议继续复用；新增 ARTrackV2 模板/搜索裁剪、400-bin 坐标解码和局部框反变换。

## 发布边界

仓库只交付 ARTrackV2 推理 runtime、球面 Geometry、Controller、比赛 I/O 和 Docker
入口。训练代码、非 ARTrack runtime、非 ARTrack 权重和非 ARTrack 校准产物不属于发布镜像。

## 数据与基线

- 测试/训练数据根目录：`E:\NewDownload\train`。
- manifest 统计：train 185153、validation 55443、calibration 10226、holdout 27450。
- 现有基线结果目录实际为 `E:\tringData\shared_control\production2`；用户提供的
  `E:\tringData\shared\_control\production2` 在当前机器不存在。
- 10 个 validation 序列按序列简单平均：
  - circular ERP mean IoU: `0.25618`
  - spherical mean IoU: `0.22360`
  - success AUC: `0.25640`

## 验证结果

- `compileall`：通过。
- ruff：通过（源码、测试和工具）。
- 全量 pytest：发布前必须在 Python 3.12 + CUDA 运行时执行；Docker build 会先验证导入图、Geometry 回归和 checkpoint。

## 权重状态

部署前需通过 Git LFS 获取 `ARTrackV2-B-256` 权重并保存为：

`models/artrackv2_b_256.pth.tar`

发布镜像固定校验该文件的 SHA-256；当前发布权重哈希记录在 `models/README.md`。
ScoreCalibration 是可选的离线产物，未提供时 ARbackendV1 使用内置 ARTrack 分数和精度优先
控制策略，不回退到其他模型。`production2` 全量对比不属于本次发布门槛。
