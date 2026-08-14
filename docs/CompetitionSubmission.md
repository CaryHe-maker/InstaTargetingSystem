# 比赛提交规范

## 构建镜像

```powershell
docker build -t instatarget:submission .
```

镜像包含 CUDA PyTorch 运行时、比赛所需项目源码、`configs/RGBonly.yaml`、内置 `src/instatarget/vendor/hit`、`models/hit_small.pth` 和无参数入口 `track.py`。本地可视化、AirSim360、训练、评估和数据工具不会进入镜像。运行主机需要 Docker Desktop、WSL 2 和 NVIDIA GPU 支持。

## 数据目录

```text
dataset/
  seqlist.txt                 # 可选
  sequence_a/
    video.mp4
    init.txt
```

`init.txt` 的前四个字段为 `clon,clat,fov_h,fov_v`，单位为角度。视频帧由持久化 OpenCV `VideoCapture` 顺序读取，BGR 转换为 RGB 后送入统一 `FramePacket`。

## 运行容器

```powershell
docker run --rm --gpus all -v "${PWD}\dataset:/mnt/dataset" -v "${PWD}\result:/mnt/result" instatarget:submission
```

入口默认读取 `/mnt/dataset`，为每个序列写出 `/mnt/result/<sequence>.txt`。可通过环境变量覆盖目录和配置路径，但比赛路由必须保持 RGB-only 配置。

## 结果格式

每帧一行 `clon,clat,fov_h,fov_v`，单位为角度，三位小数。第 0 帧写入初始 BFoV；丢失帧写四个零。帧数量、顺序和文件发布均由 `BfovResultSink` 校验。

## 视觉输出

比赛镜像不包含 visualization 模块和诊断入口，因此只会写官方 BFoV 文本结果。可视化功能保留在 GitHub 开发源码中，仅用于本地调试与评估。
