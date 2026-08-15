# Core 模块结构

Core 定义所有模块共享的值对象、协议、配置和异常，不执行模型、几何或 I/O 算法。

| 文件 | 职责 |
|---|---|
| `src/instatarget/core/types.py` | 帧、视域、观测、计划和结果类型 |
| `src/instatarget/core/protocols.py` | 模块间最小接口 |
| `src/instatarget/core/config.py` | 严格 YAML schema 和参数约束 |
| `src/instatarget/core/errors.py` | 分层异常类型 |

深入阅读：[dataTypes.md](dataTypes.md)、[transactionProtocol.md](transactionProtocol.md)、[configuration.md](configuration.md)。

