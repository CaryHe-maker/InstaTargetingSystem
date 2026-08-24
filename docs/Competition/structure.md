# Competition 模块结构

Competition 把统一跟踪管线适配为 InstaTest 的目录发现、逐序列运行和 BFoV 文本提交。

| 文件 | 职责 |
|---|---|
| `track.py` | 容器无参数入口 |
| `app/competition.py` | 视频源、序列发现、初始化和比赛 sink |
| `adapters/competition_adapter.py` | 通用结果格式适配 |
| `Dockerfile` | CUDA 提交镜像 |

深入阅读：[submissionPipeline.md](submissionPipeline.md)、[resultFormat.md](resultFormat.md)。Docker 构建要求 Git LFS 已拉取模型实体。

