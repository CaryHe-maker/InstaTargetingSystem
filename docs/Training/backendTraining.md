# 后端训练边界与建议算法

## 当前状态

训练后端已提供可执行的首版 Stage 1-4 管线。入口为 `tools/train_backend.py --config configs/train_backend.yaml`，也注册为 `trainBackend` 命令。训练仍要求 manifest 中存在人工或审核后的标签；没有标签时不能直接把视频当监督数据。

## 建议的数据对

从同一目标的可靠帧构造 template/search 对。template 应来自较稳定、遮挡较少帧；search 应覆盖运动、尺度、透视边缘和经线跨越。局部裁剪必须调用生产 Geometry，并覆盖 TRACKING 的 30°–120°动态 Type1 与 UNCERTAIN/LOST 的固定 120°视域；若训练分布刻意不同，必须明确目的并单独验证。

## 已实现损失组成

`computeTrainingLoss()` 返回 presence BCE/focal、正样本 bbox L1、GIoU 和 quality BCE 子损失。负样本不计算 bbox 损失，quality 负样本使用降低后的权重；quality target 是预测框与真值框 IoU，不使用 Controller 分数。

HiT wrapper 保留 corner head/soft-argmax，并显式输出 `presenceLogit`、`qualityLogit`、概率、预测 IoU、corner heatmap 和 object embedding。

## 训练配置

`TrainingConfig` 独立于推理 `AppConfig`，严格校验 manifest、sequence split、FOV、负样本比例、batch/累积梯度、分层学习率、AMP/BF16、warmup/scheduler、checkpoint/resume、验证周期和 early stopping。`ManifestPairDataset` 按帧索引读取视频并调用生产 Geometry 生成 template/search crop，避免从视频头部重复扫描。

训练运行会写入 `run_metadata.json`、`latest.pth`、`best.pth` 和 `final.pth`，其中保存参数组、manifest hash、初始权重 hash、配置和验证指标。

当前 `E:\NewDownload\train` 的 130 个序列具有逐帧 BFoV 真值。先运行 `tools/build_training_manifest.py` 将 BFoV 转为 seam-aware ERP bbox，并按 sim/real 分层做 sequence-level 70/15/5/10 划分。若其中包含正式测试序列，必须先用 `--exclude-file` 排除。

## 发布门槛

新权重必须通过局部跟踪验证、端到端 Controller 验证、分数校准和真实运行速度测试。checkpoint 文件名或路径变化要同步 `model.weights` 和容器构建。

