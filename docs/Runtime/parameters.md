# Runtime 参数索引

| 参数 | 当前值 | 当前状态 |
|---|---:|---|
| `runtime.decodeQueueCapacity` | 3 | 未来异步解码队列预留 |
| `runtime.inferRequestQueueCapacity` | 1 | 未来推理请求队列预留 |
| `runtime.inferResponseQueueCapacity` | 1 | 未来推理响应队列预留 |
| `runtime.resultQueueCapacity` | 32 | 未来结果队列预留 |

当前 `runTracking()` 是顺序线程，这四项只被配置校验，不会创建队列或改变延迟。并行化实现落地前，不应通过调整这些值声称提高性能。

数据路径、输出路径、sequence ID 和 instance ID 是命令参数；它们影响运行对象，但不是算法超参数。项目只有 RGB-only 运行路线。处理计时没有可调采样率，始终使用纳秒单调时钟。

