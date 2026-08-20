# 训练样本生成

## 数据集接口

`AirSim360TrainingDataset` 复用 `data.registry.openDataset()`，因此训练和运行读取完全相同的 RGB 与 mask。构造时必须显式给出 targetInstanceId，避免自动选择对象导致实验不可复现。

## 样本内容

每个 `TrainingSample` 包含原 FramePacket、targetInstanceId、targetBox 和 visible。targetBox 由 MaskPseudoTrackBuilder 的循环 ERP 算法生成，能表达跨经线目标；visible=false 时仍保持样本索引。

## Manifest 随机访问

`AirSim360TrainingDataset` 保留用于旧 mask 数据检查。正式训练使用 `ManifestPairDataset`：manifest 直接保存 `videoPath/frameIndex`，每个 worker 维护小型 VideoCapture LRU 并 seek 到目标帧，不再为一次随机样本从视频头部扫描。

## 模型输入

稳定、无遮挡帧作为 template 候选；search 按配置的 frame gap 和 30°–120° FOV 采样。两者都通过生产 `SphericalGeometryImpl` 生成局部视图，ERP bbox 边界再投影为归一化局部 `cx,cy,w,h`。数据集生成中心/边缘正样本、目标缺失负样本和 off-view 负样本，并保留 `labelSource`、`labelQuality` 与 `difficultType`。

## 必要校验

`build_training_manifest.py` 将当前 BFoV groundtruth 转成 seam-aware ERP bbox，并按 sim/real 分层执行 sequence-level train/validation/calibration/holdout 划分。加载器会拒绝同一 sequence 出现在多个 split。正式测试序列仍需通过 exclude 文件显式隔离。

