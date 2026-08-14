# 实现说明

本文汇总仓库内可执行运行线路及其边界。架构关系见 [Design.md](Design.md)，逐帧行为见 [process.md](process.md)。

## 可执行功能

- 严格 YAML 配置加载与参数约束。
- ERP 边界框、球面 BFoV、单位方向和局部透视视图转换。
- 真实 PyTorch HiT-Small 模型加载、模板编码和 CUDA 推理。
- RGB-only 单会话后端。
- RGB-D 双会话后端、深度伪彩色编码和分数融合。
- 球面运动估计、多视角候选聚类、状态评估、同帧升级和恢复搜索。
- AirSim360 RGB、深度、语义图和实例图读取。
- `.mp4` 比赛视频读取及 BFoV 结果输出。
- 开发结果、IoU 指标、中间可视化和最终结果图像输出。
- 原子结果发布与严格帧序检查。

## HiT 集成

`PyTorchHiTSession` 从 `src/instatarget/vendor/hit` 导入内置的最小 HiT-Small 运行时，读取 `configs/HiT_Small.yaml`，并加载 `models/hit_small.pth` 中的 `net` 状态字典。模板裁剪为 `128 x 128`，搜索图缩放为 `256 x 256`，输入使用 ImageNet 均值与标准差归一化。

模型置信度由角点头两个热图的归一化熵和集中度计算。FP16 前向产生非有限边界框或热图时，适配器使用同一模型执行 FP32 重算；该处理不切换模型结构或权重。

## 仓库范围

生产运行时只连接 PyTorch HiT-Small。训练目录提供 NumPy 数据样本、伪真值和损失接口，仓库不包含完整的模型训练任务。导出、替代推理后端以及通用日志和计时辅助文件不属于比赛容器执行路径。

HiT-Small 运行依赖已收敛到 `src/instatarget/vendor/hit`，只保留模型构建、配置、基础工具和 checkpoint 兼容类型。训练、评测、tracking 示例、数据脚本和上游文档不进入项目运行时或 Docker 镜像。

## 资源生命周期

运行时创建模型会话后，由 `TrackerBackend.close()` 统一释放会话、前向钩子和 CUDA 缓存。RGB-D 创建第二会话失败时，已创建的 RGB 会话会立即关闭。CLI 在正常结束和异常退出路径中都会关闭数据源与后端。
