# 生命周期与处理计时

## 计时对象

`visualization/time_counter.py::TimeCounter` 允许多次 `startProcessing()/stopProcessing()`，把每个区间累加到一个 elapsed 值。`time.json` 保持 `instatarget.time.v1`，并用 `scope=tracking_processing` 明确语义。

## 计入范围

帧读取/解码、初始化模板、Controller 计划、视图裁剪、HiT 推理、外观与运动校准、一次边界回投、SingleScore 合成、StateEvaluator、状态转移和 TrackResult 构造都计入。

## 排除范围

配置加载、组件创建、中间可视化、最终结果绘制、sink 写入/最终化、time.json 写入和资源清理不计入。`startedAtUtc` 与 `finishedAtUtc` 只是产物生命周期，二者之差不是处理耗时。

## 异常语义

处理区间用 `try/finally` 关闭，因此区间内失败仍记录已花时间。若配置阶段失败且从未开始处理，elapsed 为 0。驱动边界由 `tests/integration/test_driver_smoke.py` 的检查型 source/sink/recorder 回归测试保护。

## 未来并行化

当前值是顺序线程墙钟时间。若未来建立异步解码和推理流水线，不能简单相加各工作线程时间；远端评测通常关心端到端处理墙钟，应定义从一帧可处理到结果可提交的区间，并在文档中升级 scope/version。

