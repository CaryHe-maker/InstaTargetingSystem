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
- `firstRoundFusionOverlap < overlapThreshold`。
- `uncertainThreshold < acceptThreshold <= recoverAcceptThreshold`。
- 三轮配置的 `maxViewsPerFrameTotal` 至少容纳 4+4+6=14 个视图。
- `fusionHead` 权重非负且至少一个为正。
- DecisionGate 三项权重总和不超过 1。

## 修改流程

增加参数时同时修改配置数据类、严格字段集合、解析构造、`RGBonly.yaml`、`RGBD.yaml`、配置测试和对应算法文档。若参数已经不影响生产路径，应明确标为兼容参数，而不是让读者误以为它仍在调节算法。

