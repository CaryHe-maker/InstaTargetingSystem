# StateEvaluator 与 Fusor

## 事务候选评估

实现位于 `controller/state_evaluator.py`。TRACKING/UNCERTAIN 第一轮只把本轮 `ProjectedObservation` 交给 Fusor，最佳候选中心作为第二轮搜索中心；没有候选时回退到运动预测中心。第二轮结束时，第一轮和第二轮的全部观测合成同一事务候选池，再统一交给 Fusor。保留的显式 LOST 组件只有一轮，直接使用该轮 6 张 cubemap 和 4 张 Type1；当前正常评分线程不会进入该模式。

Runtime 先完成局部框回投和 SingleScore，再把投影结果交给 StateEvaluator。StateEvaluator 调用 Fusor 得到当前提交点的唯一最佳候选，输出 StateObservation、StateScore、测量接受资格和诊断字段。

## Fusor

实现位于 `controller/fusor.py`。Fusor 使用现有 seam-aware ERP 算法：

```text
OverlapRate = ERP 交集面积 / min(框 A 面积, 框 B 面积)
```

Fusor 先把每个观测加入单框候选，再枚举全部无序观测对。只有两个来源的 SingleScore 都不低于 `evaluator.fusionSourceMinConfidence`，且 OverlapRate >= 固定常量 0.70 时，才生成两框融合候选。三种状态使用同一套融合几何，最多使用两个来源。

OverlapRate 只负责判断两个来源是否足够重叠。融合置信度另用 seam-aware ERP IoU 衡量两框整体一致性：

```text
agreementIoU = ERP 交集面积 / ERP 并集面积
base = sqrt(a * b)
consistency = 1 - abs(a - b)
bonus = 0.15 * agreementIoU * consistency * (1 - base)
fusionScore = min(base + bonus, max(a, b) + 0.03, 0.99)
```

几何平均值抑制一高一低来源造成的虚高，IoU 奖励只在来源分数一致且框形状一致时生效；最终分数最多比最高来源增加 0.03，并硬限制在 0.99。Fusor 将所有单框和可行融合框统一排序，只返回一个最佳结果。先比较 confidence，同分时优先融合候选，再选择 representative viewId 较小者。最终测量还必须达到 `tracking.candidateMinScore`。Fusor 不会返回候选列表，也不会融合三个或更多来源。

## 统一的参考面积裁剪

`TRACKING`、`UNCERTAIN` 和 `LOST` 进入生产 StateEvaluator 后都固定使用 `REFERENCE_ADAPTIVE`，不再按状态分别选择合并框或交叉框。Controller 传入上一可信框的 ERP 面积作为 `size0`；第一个搜索帧使用 frame 0 初始化框。未接受的弱候选不会更新当前框，因此也不会改变下一帧的 `size0`。

若最高分候选是单框，Fusor 原样返回该框。若最高分候选是两框融合结果，则先计算 seam-aware 最大连通交叉框 `size1` 和最小循环合并框 `size2`，再按面积选择最终输出：

```text
size0 <= size1          -> 输出最大交叉框
size1 < size0 < size2   -> 将交叉框按中心缩放到 1.5*size0，再与合并框取交集
size0 >= size2          -> 直接输出最小合并框，不放大
```

居中缩放保持交叉框宽高比，并受 ERP 帧宽高约束；中间分支若无法得到有效交集则回退原交叉框。最终 bbox 确定后重新计算对应 BFoV。该裁剪只改变最佳融合候选的输出几何，不改变来源、融合分数或候选排名。

## 保留的 Classifier

`controller/classifier.py` 仍保留确定性的加权球面聚类实现，供实验和兼容测试使用，但当前生产 Controller 不调用它，第二轮也不依赖 classify 结果。

若单独使用该工具，输入必须是投影后的观测，聚类坐标为 `ProjectedObservation.bfov.center` 的单位球面向量。每个类中心由 SingleScore 加权，所有成员到中心的大圆距离不超过 30°；结果按成员数量、类内平均 SingleScore 和 viewId 稳定排序，最多返回 3 个中心。

## 分状态路由

`TRACKING`：第一轮在预测中心周围取 VStype1 四角 4 张；第一轮 Fusor 最佳中心周围再取 VStype1 四角 4 张。

`UNCERTAIN`：第一轮以预测中心为中心取 VStype1 四角 4 张，且每张视图为 120°×120°；第一轮 Fusor 最佳中心周围再取同样的 VStype1 四角 4 张。

`LOST`（保留组件，正常线程不可达）：第一轮一次性读取 6 张旋转 cubemap 和预测中心周围的 Type1 四角 4 张，共 10 张，统一交给 Fusor；不再追加第二轮。

正常线程中的 TRACKING/UNCERTAIN 固定执行两轮，最终 StateScore 取两轮候选统一经过 Fusor 后的最佳分数；显式 LOST 组件取单轮 10 张视图的 Fusor 最佳分数。三种模式的最佳融合候选都使用上述参考面积裁剪。没有候选时 StateScore 为 0，并输出预测框但不接受测量；状态机随后保持 UNCERTAIN。`evaluator.successRate` 只写入诊断字段，不参与当前升级、排序或提交决策。
