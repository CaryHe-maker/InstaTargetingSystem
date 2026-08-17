# 配置加载与参数约束

## 加载算法

`loadConfig()` 使用 PyYAML 读取根对象，然后逐节执行三类检查：字段集合必须精确匹配、标量类型必须正确、数值范围和跨字段关系必须成立。相对路径以配置文件所在目录解析，而不是以进程当前目录解析。

这种严格 schema 会让拼写错误立即失败。例如新增 YAML 字段但未更新 `_section()` 的允许集合，会被识别为未知字段；只修改数据类而不修改两份 YAML，则会成为缺失字段。

## 配置组归属

- `model/depth/backendFusion/fusionHead`：Tracker。
- `geometry`：Geometry 和视域输出尺寸。
- `evaluator/motion/tracking/recovery`：Controller。
- `runtime`：Runtime 的未来队列容量。
- `visualization`：诊断产物开关。

所有参数的算法作用在所属模块专题文档中说明。

## 关键交叉约束

- `geometry.maxFovDeg` 必须为 120，保证固定最大搜索视域。
- schema 仍要求 `firstRoundFusionOverlap < overlapThreshold`，但两项当前只为配置兼容；生产 Fusor 使用固定 0.70 常量。
- 状态阈值不再来自 YAML 标量；状态机根据最近 10 个 `StateScore` 动态计算 UT/LT。
- `tracking.maxAttemptsPerFrame` 固定为 2，`tracking.maxViewsPerFrameTotal` 至少容纳 12 张视图（LOST 的两个旋转 cubemap）。
- `fusionHead` 权重非负且至少一个为正。
- DecisionGate 三项权重总和不超过 1；生产 StateEvaluator 当前忽略整组 DecisionGate。

外观 Beta 参数、运动残差组合权重和 70/30 SingleScore 当前是 `controller/fused_score.py` 中的冻结代码常量，不属于 YAML。`decisionGate.*Weight` 只服务旧兼容聚合，不控制生产路径的 SingleScore。

## 修改流程

增加参数时同时修改配置数据类、严格字段集合、解析构造、`RGBonly.yaml`、`RGBD.yaml`、配置测试和对应算法文档。若参数已经不影响生产路径，应明确标为兼容参数，而不是让读者误以为它仍在调节算法。

