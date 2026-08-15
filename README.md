# InstaTargetingSystem

InstaTargetingSystem 是面向 ERP 全景视频的单目标跟踪系统，提供 RGB-only 与 RGB-D 两条运行线路、真实 HiT-Small CUDA 推理、球面运动控制、恢复搜索、AirSim360 评估和 InstaTest 比赛提交入口。

## 运行线路

- RGB-only：一个 HiT-Small 会话处理局部 RGB 视图。
- RGB-D：两个独立 HiT-Small 会话分别处理局部 RGB 与深度伪彩色视图。
- 比赛提交：持久化读取 `.mp4`，强制 RGB-only，并按帧输出 BFoV 角度结果。

生产运行使用项目内置的 `src/instatarget/vendor/hit`、`models/hit_small.pth` 和可用的 CUDA PyTorch 环境。项目不会在模型缺失时切换到本地替代会话。

## 快速命令

安装项目入口后，可直接运行：

```powershell
run -RGB_only /data/airsim360/nyc_sample /artifacts/easy_user/nyc_sample/RGB_only 14211313
run -RGBD /data/airsim360/nyc_sample /artifacts/easy_user/nyc_sample/RGBD 14211313
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
