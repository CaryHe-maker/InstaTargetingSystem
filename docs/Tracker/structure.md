# Tracker 模块结构

Tracker 管理局部视图推理、模板特征、深度表示和 RGB-D 分数融合，不负责球面搜索或状态转移。

| 文件 | 职责 |
|---|---|
| `hit_backend.py` | HiTSession 协议与异常边界 |
| `pytorch_hit_session.py` | 真实 HiT-Small 推理 |
| `backend.py` | 批量视图、RGB/RGB-D 和模板命令编排 |
| `depth_preprocessor.py` | 深度清洗、归一化、伪彩色与摘要 |
| `depth_encoder.py` | 深度会话特征适配 |
| `fusion_head.py` | RGB、深度和上下文分数融合 |
| `template.py` | 模板特征缓存和 revision |

深入阅读：[hitRuntime.md](hitRuntime.md)、[rgbdFusion.md](rgbdFusion.md)、[depthPipeline.md](depthPipeline.md)、[templateCache.md](templateCache.md)、[parameters.md](parameters.md)。
