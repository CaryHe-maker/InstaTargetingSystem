# 处理时间产物

## 累计算法

TimeCounter 用 `perf_counter_ns()` 记录每个处理区间起止，将差值累加到 `_processingElapsedNs`。它不通过 UTC 时间相减计算耗时，避免系统时钟调整影响结果。

## JSON 字段

- `format=instatarget.time.v1`：文件 schema。
- `scope=tracking_processing`：只统计跟踪处理路径。
- `elapsedNanoseconds/Milliseconds/Seconds`：同一累计值的三种单位。
- `startedAtUtc/finishedAtUtc`：产物生命周期时间戳，仅用于审计。

## 计时边界

计入帧读取、裁剪、推理、校准、投影和 Controller 计算；不计入任何可视化、sink、配置和清理。完整顺序见 `Overall/runtimeThread.md`。

## 安全状态

重复 startProcessing 或在未开始时 stopProcessing 会抛 RuntimeError，防止嵌套/漏停让数字失真。stop() 遇到仍活动区间会先关闭；没有处理区间时写 0。

## 如何比较性能

比较配置时应同时报告总 elapsed、处理帧数、每帧均值和状态/round 分布。单独比较总秒数会被不同序列长度或更多找回轮次误导。

