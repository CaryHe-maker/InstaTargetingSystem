# 比赛提交要求

本文件保留原路径，用于说明项目对 InstaTest 参考格式的适配结论。完整构建与运行规范见 [CompetitionSubmission.md](CompetitionSubmission.md)。

## 输入约定

比赛数据根目录由多个序列目录组成。每个有效序列包含一个 `.mp4` 视频和一个 `init.txt` 文件；根目录可通过 `seqlist.txt` 指定序列顺序。`init.txt` 前四个逗号分隔值依次为：

```text
clon,clat,fov_h,fov_v
```

四个值均使用角度制。经度范围为 `[-180, 180)`，纬度范围为 `[-90, 90]`，水平与垂直视场角均位于 `(0, 180)`。

## 视频与模态

`OpenCvVideoSource` 使用一个持久化的 `cv2.VideoCapture` 顺序解码视频，将 OpenCV 的 BGR 帧转换为连续内存的 RGB 数组，并为每帧生成严格递增的 `frameIndex`。InstaTest 路线只提供色彩视频，不读取深度信息。

官方提交入口强制使用 RGB-only 配置：

- `depth.enabled` 必须为 `false`。
- `backendFusion.depthScoreWeight` 必须为 `0.0`。
- 每个序列只创建一个真实 HiT-Small 会话。

## 输出约定

每个输入帧对应一行 BFoV 结果，顺序与输入帧一致：

```text
clon,clat,fov_h,fov_v
```

结果使用角度制并保留三位小数。第 0 帧写入 `init.txt` 给定的初始 BFoV；无有效观测的帧写为：

```text
0.000,0.000,0.000,0.000
```

写入过程先生成 `.partial` 文件，帧数核对成功后再原子替换最终结果文件。

## 容器入口

根目录 `track.py` 是无参数容器入口，从环境变量读取：

| 变量 | 默认值 |
|---|---|
| `DATASET_DIR` | `/mnt/dataset` |
| `RESULT_DIR` | `/mnt/result` |
| `CONFIG_PATH` | `/app/configs/RGBonly.yaml` |
| `HIT_ROOT` | `/app/third_party/HiT` |

可视化记录器不参与比赛结果计算。启用或关闭可视化不会改变 HiT 输入、控制器状态、候选分数或 BFoV 输出，因此不影响测试逻辑；启用后只会增加磁盘写入和运行时间。
