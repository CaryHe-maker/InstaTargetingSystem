# 运行时与依赖

## 本地运行

项目需要 Python 3.11、CUDA、PyTorch 2.6、torchvision 0.21、OpenCV、PyYAML、NumPy、timm 和 EasyDict。依赖清单位于 `requirements.txt`，开发测试依赖位于 `pyproject.toml`。

真实 HiT-Small 运行需要：

- `src/instatarget/vendor/hit` 内置的最小 HiT-Small 运行时；
- `models/hit_small.pth` 权重；
- 可用的 CUDA PyTorch 运行时。

适配器默认从源码包内查找 HiT 运行时，也接受 `HIT_ROOT` 环境变量。模型配置中的权重路径相对于配置文件解析。

## 线路选择

`configs/RGBonly.yaml` 关闭深度并创建一个 HiT 会话。`configs/RGBD.yaml` 启用深度并创建两个独立 HiT 会话，第二个会话处理深度伪彩色图。FP16 非有限输出会在同一模型上自动执行 FP32 重算。

## Docker

Dockerfile 使用 `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`，复制经过 `.dockerignore` 筛选的比赛源码、RGB-only 配置、内置 HiT 运行时、权重和 `track.py`。容器通过 `DATASET_DIR`、`RESULT_DIR`、`CONFIG_PATH` 和 `HIT_ROOT` 指定输入、输出、配置和模型源。

## 资源释放

应用入口在正常和异常路径关闭视频源、结果 sink 和后端。HiT 关闭过程移除热图钩子并释放 CUDA 缓存。模型权重保存在 `models/hit_small.pth`，应用配置位于 `configs/*.yaml`，HiT-Small 网络配置位于 `src/instatarget/vendor/hit/configs/HiT_Small.yaml`。
