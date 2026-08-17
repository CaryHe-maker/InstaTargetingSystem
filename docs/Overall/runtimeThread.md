# 完整运行线程

本文描述一次 AirSim360 运行从进程启动到结果关闭的实际顺序。核心实现位于 `src/instatarget/app/track_airsim360.py` 和 `src/instatarget/app/driver.py`。

## 进程启动

1. CLI 解析配置路径、数据根、目标 instance ID 和输出目录。
2. `loadConfig()` 严格读取 YAML；未知键和缺失键都会失败。
3. AirSim360 source 打开序列并读取元数据；伪真值构建器从第 0 帧实例掩码得到初始化框。
4. `buildRuntime()` 创建 Geometry、Controller、Tracker backend、可选 DepthProcessor、sink 和可选中间可视化 recorder。
5. `TimeCounter.start()` 只建立时间产物生命周期，此时还没有累计处理时间。

## 初始化帧

`runTracking()` 开启第一个处理区间，然后：

1. 读取并解码第 0 帧。
2. Controller 将 ERP 初始化框转换为模板视域和局部模板框。
3. Geometry 裁剪模板 LocalView。
4. Tracker 编码初始模板；RGB-D 模式同时编码深度模板。
5. Controller 初始化运动历史和 `TRACKING` 状态，生成第 0 帧 `TrackResult`。

处理区间在结果构造完成后关闭。随后 sink 写第 0 帧、最终结果可视化和中间模板视图，这些都不计入处理时间。

## 普通帧

每帧开始一个新的处理区间：

1. Source 读取一帧。若为 EOF，本轮不产生结果并结束循环。
2. `controller.beginFrame()` 根据可靠历史预测中心，建立 `FrameTransaction` 和 Round 1 `SearchPlan`。
3. Geometry 按计划一次裁剪本轮所有视图。
4. Tracker 按本轮 view 顺序执行批量局部推理并返回 LocalObservation。PyTorch HiT 每轮做一次 RGB tensor forward；RGB-D 再对有深度的视图做一次 depth tensor forward。
5. Beta Calibration 生成 `appearanceProbability`，保留 backend 原始融合分。
6. Geometry 将每个局部框边界一次回投，直接拟合 ERP bbox 和紧致 BFoV，并保留边界/膨胀诊断。
7. Runtime 按局部 ViewSpec 中心与运动预测中心的大圆夹角生成当前生产运动概率，并按 70/30 合成 SingleScore；协方差归一化候选残差保留为离线诊断路径。
8. `controller.consume()` 调用 StateEvaluator。TRACKING/UNCERTAIN 第一轮用 Fusor 最佳候选中心生成第二轮 VStype1 四角计划，没有候选则回退到预测中心；第二轮完成后把两轮投影观测统一交给 Fusor。返回 `FrameCommitted` 后结束本帧处理。
9. 处理区间关闭后，Runtime 才写中间可视化、sink 和最终结果可视化。

所以同一输入帧只读取一次。TRACKING 使用 4+4、UNCERTAIN 使用 6+4，均执行两轮批量局部推理；LOST 单轮批量处理 12 张。轮次不能合并，因为第二轮中心依赖第一轮 Fusor 结果。所有轮次属于同一个 FrameTransaction，只有最后一次 consume 能提交持久状态。

## 结束和异常

正常 EOF 后 sink 校验结果帧数并原子发布文件。无论正常或异常，source、backend 和 sink 都在生命周期末尾关闭，`time.json` 最后写入。如果处理区间内抛错，`finally` 仍会停止该区间，因此已消耗的处理时间不会丢失。

## 线程与队列说明

当前主路径是单进程顺序线程，不会按 `runtime.*QueueCapacity` 创建真正的异步队列；批量 HiT 只并行化同一轮的 GPU 张量计算。配置中的队列容量是未来流水线并行化的预留字段。优化并行度时必须保持三条顺序约束：模板 revision 有序、同帧轮次有序、FrameTransaction 只能提交一次。

