# 运行时与依赖

## 本地运行

项目需要 Python 3.11、CUDA、PyTorch 2.6、torchvision 0.21、OpenCV、PyYAML、NumPy、timm、EasyDict 和 tensorboardX。依赖清单位于 `requirements.txt`，开发测试依赖位于 `pyproject.toml`。

真实 HiT-Small 运行需要：

- `third_party/HiT` 官方源代码树；
- `models/hit_small.pth` 权重；
- 可用的 CUDA PyTorch 运行时。

适配器默认从仓库相对路径查找 HiT 源树，也接受 `HIT_ROOT` 环境变量。模型配置中的权重路径相对于配置文件解析。

## 线路选择

`configs/RGBonly.yaml` 关闭深度并创建一个 HiT 会话。`configs/RGBD.yaml` 启用深度并创建两个独立 HiT 会话，第二个会话处理深度伪彩色图。FP16 非有限输出会在同一模型上自动执行 FP32 重算。

## Docker

Dockerfile 使用 `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`，复制 `src`、RGB-only 配置、HiT 源码、权重和 `track.py`。容器通过 `DATASET_DIR`、`RESULT_DIR`、`CONFIG_PATH` 和 `HIT_ROOT` 指定输入、输出、配置和模型源。

## 资源释放

应用入口在正常和异常路径关闭视频源、结果 sink 和后端。HiT 关闭过程移除热图钩子并释放 CUDA 缓存。模型权重和参数保存在 `models/hit_small.pth` 与 `configs/*.yaml`，不保存在 `third_party/HiT` 文档中。
