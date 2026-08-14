# 比赛提交规范

## 构建镜像

```powershell
docker build -t instatarget:submission .
```

镜像包含 CUDA PyTorch 运行时、项目源代码、`configs/RGBonly.yaml`、`third_party/HiT`、`models/hit_small.pth` 和无参数入口 `track.py`。运行主机需要 Docker Desktop、WSL 2 和 NVIDIA GPU 支持。

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

比赛默认关闭可视化。可视化只写诊断 PNG，不改变模型或控制器，因此不会影响官方测试结果；开启后会增加存储和运行时间。
