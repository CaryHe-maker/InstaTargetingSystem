# 运行环境

## 基础要求

- Python 3.11 或兼容版本。
- NVIDIA GPU、可用驱动和 CUDA PyTorch 2.6 运行时。
- Docker Desktop 使用 WSL 2 后端时，需要启用 GPU 容器支持。
- `src/instatarget/vendor/hit` 内置的最小 HiT-Small 运行时。
- `models/hit_small.pth` HiT-Small 权重。

Python 依赖由 `requirements.txt` 和 `pyproject.toml` 定义，包括 NumPy、OpenCV、PyYAML、PyTorch、torchvision、timm 和 EasyDict。开发检查额外使用 pytest 与 Ruff。

## 本地验证

```powershell
& ".venv\Scripts\python.exe" -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
& ".venv\Scripts\python.exe" -m pip check
& ".venv\Scripts\python.exe" -m pytest -q
```

`PyTorchHiTSession` 只在 CUDA 可用时创建。运行时查找顺序为显式 `hitRoot`、`HIT_ROOT` 环境变量和源码包内 `src/instatarget/vendor/hit`。权重路径由 YAML 配置解析。

## Docker

根目录 Dockerfile 基于 `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`，并将真实 HiT 源码和权重复制到镜像。构建与比赛运行命令见 [CompetitionSubmission.md](CompetitionSubmission.md)。

## 模态配置

`configs/RGBonly.yaml` 使用 FP32 单会话。`configs/RGBD.yaml` 使用 FP16 双会话；检测到非有限模型输出时，同一模型以 FP32 重算。比赛入口只接受 RGB-only 配置。
