# StateEvaluator 候选算法

实现位于 `controller/state_evaluator.py`。输入是同一轮多个 ProjectedObservation，输出是完整 `StateObservation`，供加轮、状态机、结果和诊断共同使用。

## 候选构造

每个投影观测先形成一个单框候选。随后只在不同 viewId 之间计算 OverlapRate：

```text
OverlapRate = ERP交集面积 / min(框A面积, 框B面积)
```

横向交集使用循环 ERP 区间，能识别跨左右边界的同一目标。

## 一对一融合

所有达到本轮融合阈值的边按 OverlapRate 降序，再按来源分数和 viewId 稳定排序。贪心选边时，一个来源框一旦使用就不能再次融合。因此融合框永远只有两个来源；一个框同时匹配多个框时只选择重合率最高的一个。

融合几何取两个 ERP 框的交叉区域，不取最小外包框。若循环交集在经线处被拆成两段，算法先合并首尾相邻段；无法表达多个断开区域时保留最大连通交集。

## 融合分数

设两来源置信度为 a、b，OverlapRate 为 y：

```text
fuseScore = 1 - ((2 - a - b) * (1 - y) / 2)
```

此外记录 `minSourceConfidence=min(a,b)`。融合分数高并不自动可靠：两个来源都必须达到 `fusionSourceMinConfidence`。

## 分轮阈值

非 LOST 的 Round 1 使用 `firstRoundFusionOverlap=0.30` 生成融合候选，但只有 y 大于 `overlapThreshold=0.70` 的融合框才能成为可靠输出。Round 2、Round 3 和 LOST Round 1 从一开始只融合 y 大于 0.70 的候选。

## 选择与加轮

候选先按置信度排序，同分时融合候选优先，再按 representative viewId 稳定选择。Round 1 只有可靠融合可以提前提交；UNCERTAIN/RECOVERING Round 2 的最高候选超过 `successRate` 可提交；各状态最终轮总是选择最高候选，不再因分数不足增加轮次。

`searchSeedCenter` 永远取当前最佳候选中心；没有候选时回退到预测中心。`StateObservation` 同时保存阈值、来源 viewId、是否融合、OverlapRate、分数分解和拒绝原因，用于定位阈值问题。

