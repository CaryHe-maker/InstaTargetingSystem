# InstaTargetingSystem

InstaTargetingSystem 是面向 ERP 全景视频的 RGB-only 单目标跟踪系统，提供真实 HiT-Small CUDA 推理、球面运动控制、恢复搜索、AirSim360 评估和 InstaTest 比赛提交入口。

## 运行线路

- 本地与比赛共用一条 RGB-only 路线：一个 HiT-Small 会话处理局部 RGB 视图。
- 比赛提交持久化读取 `.mp4`，并按帧输出 BFoV 角度结果。

生产运行使用项目内置的 `src/instatarget/vendor/hit`、`models/hit_small_stage3_inference.pth`、与其哈希绑定的 `models/hit_small_stage3_inference.calibration.json` 和可用的 CUDA PyTorch 环境。项目不会在模型或校准产物缺失时切换到替代路径。

GitHub 已包含小于 100 MB 的压缩 Stage 3 checkpoint 及其哈希绑定校准文件；国内构建服务器 clone 后可直接执行 `python docker/verify_submission.py` 和 `docker build`。提交镜像固定基于 `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel`，包含 RTX 5090 D v2 所需的 `sm_120` 支持；当前实现为 7 个 RootFS layer，低于最多 10 层的限制。具体流程见 [提交容器运行环境](docs/Competition/containerRuntime.md)。

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
- [HiT 运行时](docs/Tracker/structure.md)
- [比赛提交](docs/Competition/structure.md)
- [运行环境与编排](docs/Runtime/structure.md)
- [验证与评估](docs/Evaluation/structure.md)

`docs/Prepare/` 保存设计讨论材料，不作为交付实现规范。
