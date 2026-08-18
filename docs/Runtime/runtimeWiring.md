# 运行组件装配

## `buildRuntime()` 的决策

Runtime 根据 AppConfig 创建一个 `RuntimeBundle`。Geometry 总是存在；Controller 总是使用同一套状态机。模型 backend 根据 `model.backend` 选择会话类型；当前生产路径完整支持 PyTorch HiT，ONNX/TensorRT 是适配边界。

Runtime 只创建一个 RGB HiT 会话并交给 TrackerBackendImpl。Visualization 关闭时 recorder 为 `None`，因此不会创建输出目录。

## 依赖注入

`hitSessionFactory` 可由测试替换，使驱动集成测试不需要真实 CUDA 权重。Controller 只接收 Geometry、MotionEstimator、StateEvaluator 等接口，不知道具体 HiT 类。

## CLI 路径

`run` 命令默认使用 `configs/RGBonly.yaml`，也允许通过 `--config` 指定兼容 schema 的配置，解析用户路径后转交 AirSim360 入口。`getInstanceID` 只读取第一帧分割，不建立模型或 Controller。

## 优化注意

组件应在序列外复用还是每序列重建取决于模板状态。当前 backend/template/controller 都携带序列状态，因此比赛多序列运行为每个序列建立独立 runtime，防止模板和运动历史串序列。

