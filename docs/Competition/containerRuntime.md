# 提交容器运行环境

## 镜像内容

根目录唯一的 Dockerfile 打包比赛所需源码、`configs/RGBonly.yaml`、内置 HiT vendor、推理权重和无参数 `track.py`。开发工具、测试、文档、数据、可视化、缓存和原始训练 checkpoint 不进入构建上下文。

基础镜像固定为 `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel`，已经提供 Python 3.12、PyTorch 2.11、torchvision 0.26、CUDA 12.8 和 RTX 5090 D v2 所需的 `sm_120` 编译架构，容器依赖清单不会重复安装 torch 和 torchvision。该组合符合评测服务器 NVIDIA driver 580、CUDA 最高 12.8、单卡 24 GB 显存、16 CPU 和 64 GB 内存的约束。构建保留预编译 PyTorch 与 NVIDIA wheel 运行库，移除只用于开发的 PyTorch 源码/Git 历史、CUDA 编译工具链、Nsight、头文件、Triton、CMake 和测试资源；competition 路线不在容器内编译扩展。随后把清理后的有效文件系统复制进 `scratch` 最终阶段，使删除内容不会残留在基础镜像层中。

## 紧凑权重

原始 `models/hit_small_stage3.pth` 包含 Stage 3 训练状态。维护者替换原始权重后运行：

```powershell
.\.venv\Scripts\python.exe docker\compact_checkpoint.py
```

脚本逐张量验证后生成 `models/hit_small_stage3_inference.pth`，只保留完全相同的 Stage 3 `model` 状态；同时生成 `hit_small_stage3_inference.calibration.json`，其 checkpoint 哈希绑定压缩文件，其他拟合参数与工作点保持不变。Docker 按原文件名复制到 `/app/models/`，与 `configs/RGBonly.yaml` 完全一致；原训练 checkpoint 不会进入构建上下文。

## 默认挂载

- 数据：`/mnt/dataset`
- 结果：`/mnt/result`
- 配置：`configs/RGBonly.yaml`

入口可以通过约定环境变量覆盖路径，但不能改变官方序列和结果格式。

## 构建与运行

最终提交流程以 GitHub checkout 为唯一输入。压缩 Stage 3 权重和哈希绑定校准文件已纳入版本控制，国内服务器不需要访问开发机 checkpoint，也不应在服务器重新运行 `compact_checkpoint.py`：

```powershell
git clone <repository-url>
cd InstaTargetingSystem
python docker/verify_submission.py
docker build -t instatarget:submission .
python docker/verify_submission.py --image instatarget:submission
docker run --rm --gpus all -v "${PWD}\dataset:/mnt/dataset" -v "${PWD}\result:/mnt/result" instatarget:submission
```

Dockerfile 的最终 `scratch` 阶段使用 7 条文件系统 `COPY`，分别合并 `/layer-parts/00` 至 `06`，低于最多 10 个 RootFS layer 的提交限制。`ENV` 和 `ENTRYPOINT` 只写镜像配置，不产生 RootFS layer。验证脚本在构建前检查必需文件已被 Git 跟踪且没有被 `.dockerignore` 排除，并核对 checkpoint/calibration 哈希、固定基础镜像和静态 7 层结构；Docker 构建阶段会断言 PyTorch、torchvision、CUDA 和 `sm_120` 版本，并在 CPU 上构造 HiT-Small、严格加载全部 checkpoint 参数。构建后验证还会在无网络容器中重复版本、架构和模型探针，并检查 `.RootFS.Layers` 数量在 1 至 10 之间、镜像 `.Size` 不超过 5,000,000,000 bytes。任一文件、参数、运行时模块、CUDA 架构、层数或体积偏差都会失败。

构建完成后可用 `docker image inspect instatarget:submission --format='{{.Size}}'` 检查总字节数，并用 `docker history instatarget:submission` 检查层大小。上传镜像时不应包含原始 checkpoint、构建缓存或本地数据。

## 可重复性

提交前应在无本地源码挂载、无开发缓存的全新容器中运行。权重必须在镜像可见位置，不能依赖宿主机绝对路径或联网下载。若修改 Python 依赖，需要同时验证 CUDA 架构、模型加载和最终镜像大小。

## 诊断边界

比赛镜像只需要官方结果；中间可视化和本地评估工具可以留在仓库开发环境，不应成为提交成功的必要依赖。

