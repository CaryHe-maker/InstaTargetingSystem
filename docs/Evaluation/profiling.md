# 性能统计

## RuntimeProfiler

`eval/profiler.py::RuntimeProfiler` 用 `track(name)` context manager 或 `record(name, elapsedNs)` 收集任意命名代码段。每个 name 保存 count、total、min、max、mean 和 P50/P95/P99；同一帧内的重复阶段还会先求和，再生成逐帧 P50/P95/P99。

manifest 评估的 `--profile` 会把逐帧阶段写入 `*.timings.jsonl`，并在单序列 summary 中保存逐调用 `profiler`、逐帧 `profilerPerFrame` 和 `profilerMetadata`。阶段包括 decode、crop、preprocess、host-to-device、CUDA Event forward、projection、calibration、Controller、backend 和 total；每帧同时保存 batch size、forward count 和每轮 backend 元数据。

关闭 profiler 时 context manager 不读取时钟，PyTorch 会话不创建 CUDA Event，也不会为计时强制同步。100 帧 validation A/B 的所有候选与最终 TrackResult 为零差异，off/on P95 为 `313.888/313.861 ms`，差异处于运行噪声内。

## TimeCounter 区别

TimeCounter 是正式运行产物，记录整个 tracking_processing 区间的端到端墙钟；RuntimeProfiler 是开发诊断，可以重叠、嵌套或只测一个函数。各 profiler 段之和可能因为嵌套而大于端到端时间。

## 推荐分解

优化时建议至少使用：frame_decode、view_crop、rgb_infer、appearance_calibration、boundary_projection、motion_scoring、state_evaluation。每项同时按状态和 attemptIndex 分组，并记录 batch size 和模型 forward 数。当前正常线程只有 TRACKING 4+4 和 UNCERTAIN 4+4，不存在 Round 3；显式测试保留 LOST 组件时为单轮 10。GPU 报告还应包含 images/s、利用率与峰值显存，否则一次大 batch 和多次小 batch 的成本无法比较。

## GPU 同步

如果测量 CUDA kernel，普通 Python perf_counter 可能只记录异步提交时间。当前 `cudaForward` 使用 CUDA Event 并在 profiler 开启时同步结束 event；host-to-device 在 non-blocking 实验中只为 profiler 同步。正式端到端 TimeCounter 通常会在取回 bbox/heatmap 输出时自然等待，但阶段报告仍明确区分 CUDA Event forward 与 Python backend 墙钟。

`profilerMetadata` 记录 GPU 名称、CUDA/PyTorch 版本、峰值 allocated/reserved 显存、结束温度、OOM 数和 FP16 fallback 数。实验顶层 `experiment` 还记录配置/checkpoint/calibration SHA-256、Git commit、Python/平台和所有优化开关。

