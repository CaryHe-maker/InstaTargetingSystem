# 后端训练边界与建议算法

## 当前状态

`training/losses.py`、`training/train_backend.py` 和 `configs/train_backend.yaml` 目前只有 TODO，没有可执行训练循环、损失实现或超参数。运行时使用现有 `models/hit_small.pth`，不能把占位文件描述成已经支持重新训练。

## 建议的数据对

从同一目标的可靠帧构造 template/search 对。template 应来自较稳定、遮挡较少帧；search 应覆盖运动、尺度、透视边缘和经线跨越。局部裁剪必须调用生产 Geometry，并覆盖 TRACKING 的 30°–120°动态 Type1 与 UNCERTAIN/LOST 的固定 120°视域；若训练分布刻意不同，必须明确目的并单独验证。

## 建议损失组成

HiT 主干通常需要分类/heatmap 损失与 bbox 回归损失；RGB-D 若继续使用双会话结构，可分别训练 RGB 和深度伪彩色会话，再在独立数据上拟合 FusionHead 或校准参数。不要让 StateEvaluator 的融合公式反向充当网络训练标签。

## 训练配置应包含

数据清单、序列级划分、batch size、学习率与调度、epoch、优化器、随机种子、增强、checkpoint、验证频率、精度模式和恢复训练。新增这些字段时应建立独立 TrainingConfig，不要塞入推理 AppConfig。

## 发布门槛

新权重必须通过局部跟踪验证、端到端 Controller 验证、分数校准和真实运行速度测试。checkpoint 文件名或路径变化要同步 `model.weights` 和容器构建。

