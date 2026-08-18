# Tracker 模块结构

Tracker 管理局部 RGB 视图推理和模板特征，不负责球面搜索或状态转移。

| 文件 | 职责 |
|---|---|
| `hit_backend.py` | HiTSession 协议与异常边界 |
| `pytorch_hit_session.py` | 真实 HiT-Small 推理 |
| `backend.py` | 批量 RGB 视图和模板命令编排 |
| `template.py` | 模板特征缓存和 revision |

深入阅读：[hitRuntime.md](hitRuntime.md)、[templateCache.md](templateCache.md)、[parameters.md](parameters.md)。
