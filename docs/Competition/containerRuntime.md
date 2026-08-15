# 提交容器运行环境

## 镜像内容

Dockerfile 打包项目源代码、`configs/RGBonly.yaml`、内置 HiT vendor、模型权重和无参数 `track.py`。运行环境需要兼容 CUDA 的 PyTorch、torchvision、timm、OpenCV 和 PyYAML。

## 默认挂载

- 数据：`/mnt/dataset`
- 结果：`/mnt/result`
- 配置：`configs/RGBonly.yaml`

入口可以通过约定环境变量覆盖路径，但不能改变官方序列和结果格式。

## 构建与运行

```powershell
docker build -t instatarget:submission .
docker run --rm --gpus all -v "${PWD}\dataset:/mnt/dataset" -v "${PWD}\result:/mnt/result" instatarget:submission
```

## 可重复性

提交前应在无本地源码挂载、无开发缓存的全新容器中运行。权重必须在镜像可见位置，不能依赖宿主机绝对路径或联网下载。若修改 Python 依赖，需要同时验证 CUDA 架构、模型加载和最终镜像大小。

## 诊断边界

比赛镜像只需要官方结果；中间可视化和本地评估工具可以留在仓库开发环境，不应成为提交成功的必要依赖。

