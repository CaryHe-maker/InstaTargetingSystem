# 训练样本生成

## 数据集接口

`AirSim360TrainingDataset` 复用 `data.registry.openDataset()`，因此训练和运行读取完全相同的 RGB 与 mask。构造时必须显式给出 targetInstanceId，避免自动选择对象导致实验不可复现。

## 样本内容

每个 `TrainingSample` 包含原 FramePacket、targetInstanceId、targetBox 和 visible。targetBox 由 MaskPseudoTrackBuilder 的循环 ERP 算法生成，能表达跨经线目标；visible=false 时仍保持样本索引。

## 迭代与随机访问

`__iter__` 打开一个 source 顺序读取，适合流式训练；`__getitem__` 为取得某个索引会从头读到该帧，正确但不适合高频随机采样。真正训练前应增加索引缓存或预构建 manifest，而不是直接用当前随机访问实现大规模 shuffle。

## 模型输入尚缺的步骤

当前样本仍是 ERP 全景和 ERP bbox，没有自动生成 HiT template/search 对、增强、正负样本或局部视域。训练管线应复用 Geometry 创建与运行一致的 LocalView，避免训练/推理投影差异。

## 必要校验

训练前应统计目标可见率、bbox 尺寸/跨缝比例和序列分布，并按序列划分 train/validation，不能把同一序列相邻帧拆到两侧。

