# 性能统计

## RuntimeProfiler

`eval/profiler.py::RuntimeProfiler` 用 `track(name)` context manager 或 `record(name, elapsedNs)` 收集任意命名代码段。每个 name 保存 count、total、min、max，并计算 mean。

它适合拆分 crop、infer、projection、evaluation 等内部成本，不自动写 `time.json`，也不定义远端测试计时边界。

## TimeCounter 区别

TimeCounter 是正式运行产物，记录整个 tracking_processing 区间的端到端墙钟；RuntimeProfiler 是开发诊断，可以重叠、嵌套或只测一个函数。各 profiler 段之和可能因为嵌套而大于端到端时间。

## 推荐分解

优化时建议至少使用：frame_decode、view_crop、rgb_infer、depth_infer、appearance_calibration、boundary_projection、motion_scoring、state_evaluation。每项同时按状态和 attemptIndex 分组，并记录 batch size 和模型 forward 数。当前正常线程只有 TRACKING 4+4 和 UNCERTAIN 4+4，不存在 Round 3；显式测试保留 LOST 组件时为单轮 10。GPU 报告还应包含 images/s、利用率与峰值显存，否则一次大 batch 和多次小 batch 的成本无法比较。

## GPU 同步

如果测量 CUDA kernel，普通 Python perf_counter 可能只记录异步提交时间。需要在测量边界显式同步 GPU，或使用 CUDA event；批量 forward 尤其不能把异步 enqueue 时间当作完整推理时间。正式端到端 TimeCounter 通常会在取回 bbox/heatmap 输出时自然等待，但文档和报告仍应注明同步方式。

