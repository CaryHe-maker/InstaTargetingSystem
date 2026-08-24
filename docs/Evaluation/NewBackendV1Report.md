# NewBackendV1 提交报告

## 提交范围

- 基线：`main` / `PostTrainV2.4`，提交 `65d7da4`。
- 目标分支：`NewBackendV1`。
- 后端：将原 HiT/HiViT 运行链路完整替换为官方 `ARTrackV2-B-256`。
- 依赖：保持 PostTrainV2.4 的 PyTorch、torchvision、timm、NumPy、OpenCV、PyYAML 和 easydict 版本约束。
- Controller、球面 Geometry、I/O 和结果协议继续复用；新增 ARTrackV2 模板/搜索裁剪、400-bin 坐标解码和局部框反变换。

## 清理内容

本分支不保留旧 HiT/HiViT 运行实现、vendor、checkpoint、calibration、训练 wrapper、
旧后端测试和旧模型专用文档。源码、配置、Docker 配置和工具中的旧模型引用已清除。

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
- 核心模块导入：通过。
- 新增 ARTrackV2 backend 协议 smoke test：通过。
- 全量单元测试：当前工作区绑定 Python 缺少 `PyYAML` 和 `torch`，因此配置/ GPU 相关测试无法执行；这不是代码运行时依赖声明缺失，项目依赖文件仍已声明对应版本。

## 权重状态

官方 checkpoint 尚未放入仓库。部署前需下载 `ARTrackV2-B-256` 权重并保存为：

`models/artrackv2_b_256.pth.tar`

放入权重后，必须在上述 validation/holdout 数据上重新拟合 ARTrackV2 专用 score calibration，
再进行端到端 IoU 对比；本报告不虚构尚未完成的 ARTrackV2 指标。
