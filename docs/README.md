# 文档索引

本目录描述 InstaTargetingSystem 的已交付实现、数据契约、比赛适配和验证方式。`docs/Prepare/` 保留设计资料与讨论记录，不属于本索引的规范性实现说明。

## 使用与提交

- [User/README.md](User/README.md)：用户入口与四份操作说明。
- [CompetitionSubmission.md](CompetitionSubmission.md)：InstaTest 输入、输出、容器和 RGB-only 约束。
- [Verification.md](Verification.md)：测试命令、结果目录和已验证指标。
- [environment.md](environment.md)：本地 CUDA、依赖和 Docker 环境。

## 架构与运行时

- [Design.md](Design.md)：模块边界和同步数据流。
- [HiTRuntime.md](HiTRuntime.md)：真实 HiT-Small 加载、推理和 RGB-D 会话。
- [process.md](process.md)：逐帧事务流程。
- [interface.md](interface.md)：核心协议和比赛接口。
- [implement.md](implement.md)：可执行功能和仓库范围。
- [hyperparameters.md](hyperparameters.md)：配置字段与约束。

## 数据与模块

- [data.md](data.md)：`FramePacket`、深度、分割和结果契约。
- [Airsim360DataSolution.md](Airsim360DataSolution.md)：AirSim360 读取与评估。
- [InstanceID.md](InstanceID.md)：实例 ID 与初始化框规则。
- [modules/controller.md](modules/controller.md)：控制器事务与状态。
- [modules/geometry.md](modules/geometry.md)：球面和 ERP 几何。
- [modules/tracker.md](modules/tracker.md)：HiT 后端与融合。
- [modules/visualization.md](modules/visualization.md)：诊断图像阶段。
- [ModelTraning.md](ModelTraning.md)：训练数据接口边界。
- [reference.md](reference.md)：直接依赖和背景资料。
- [style.md](style.md)：工程实现规范。
