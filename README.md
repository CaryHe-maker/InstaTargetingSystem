# InstaTargetingSystem

InstaTargetingSystem 是面向 ERP 全景视频的 RGB-only 单目标跟踪系统，提供真实 ARTrackV2-B-256 CUDA 推理、球面运动控制、恢复搜索、AirSim360 评估和 InstaTest 比赛提交入口。

## 运行线路

- 本地与比赛共用一条 RGB-only 路线：一个 ARTrackV2-B-256 会话处理局部 RGB 视图。
- 比赛提交持久化读取 `.mp4`，并按帧输出 BFoV 角度结果。

生产运行使用项目内置的 ARTrackV2 vendor runtime、`models/artrackv2_b_256.pth.tar` 和 CUDA PyTorch 环境。

模型 checkpoint 不随源码提交；下载官方 ARTrackV2-B-256 权重后放入 `models/`。依赖版本沿用 PostTrainV2.4（PyTorch 2.11、torchvision 0.26、timm 0.5.4），具体容器配置见 [Dockerfile](Dockerfile)。

## 快速命令

安装项目入口后，可直接运行：

```powershell
run /data/airsim360/nyc_sample /artifacts/easy_user/nyc_sample/RGB_only 14211313
getInstanceID /data/airsim360/nyc_sample /artifacts/easy_user/nyc_sample/InstanceID.txt
```

完整评估、IoU 和可视化产物使用 `tools/run_airsim360_dataset.py`，具体命令见 [快速使用](docs/User/EasyUser.md)。

## 文档

- [文档索引](docs/Overall/README.md)
- [系统设计](docs/Overall/structure.md)
- [ARTrackV2 运行时](docs/Tracker/structure.md)
- [比赛提交](docs/Competition/structure.md)
- [运行环境与编排](docs/Runtime/structure.md)
- [验证与评估](docs/Evaluation/structure.md)

`docs/Prepare/` 保存设计讨论材料，不作为交付实现规范。
