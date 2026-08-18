# 提交容器运行环境

## 镜像内容

根目录唯一的 Dockerfile 打包比赛所需源码、`configs/RGBonly.yaml`、内置 HiT vendor、推理权重和无参数 `track.py`。开发工具、测试、文档、数据、可视化、缓存和原始训练 checkpoint 不进入构建上下文。

基础镜像已经提供匹配 CUDA 的 PyTorch 和 torchvision，容器依赖清单不会重复安装它们。构建先清理 Conda/Pip 缓存、头文件、静态库和测试资源，再把有效文件系统复制进 `scratch` 最终阶段，使删除内容不会残留在基础镜像层中。

## 紧凑权重

原始 `models/hit_small.pth` 包含 optimizer、stats 和训练设置，约 130.8 MiB（137 MB）。构建前运行：

```powershell
.\.venv\Scripts\python.exe docker\compact_checkpoint.py
```

脚本逐张量验证后生成约 46.1 MiB 的 `models/hit_small_inference.pth`，只保留完全相同的 `net` 状态。Docker 将它复制为容器内的 `/app/models/hit_small.pth`；原训练 checkpoint 不会进入构建上下文。

## 默认挂载

- 数据：`/mnt/dataset`
- 结果：`/mnt/result`
- 配置：`configs/RGBonly.yaml`

入口可以通过约定环境变量覆盖路径，但不能改变官方序列和结果格式。

## 构建与运行

```powershell
.\.venv\Scripts\python.exe docker\compact_checkpoint.py
docker build -t instatarget:submission .
docker run --rm --gpus all -v "${PWD}\dataset:/mnt/dataset" -v "${PWD}\result:/mnt/result" instatarget:submission
```

构建完成后用 `docker image inspect instatarget:submission --format='{{.Size}}'` 检查总字节数，并用 `docker history instatarget:submission` 检查层大小。提交时不应包含原始 checkpoint、构建缓存或本地数据。

## 可重复性

提交前应在无本地源码挂载、无开发缓存的全新容器中运行。权重必须在镜像可见位置，不能依赖宿主机绝对路径或联网下载。若修改 Python 依赖，需要同时验证 CUDA 架构、模型加载和最终镜像大小。

## 诊断边界

比赛镜像只需要官方结果；中间可视化和本地评估工具可以留在仓库开发环境，不应成为提交成功的必要依赖。

