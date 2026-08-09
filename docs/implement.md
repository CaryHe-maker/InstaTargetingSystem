# InstaTargetingSystem 实现清单

> 目标：保留已完成的 geometry、RGB-only、RGB-D 和 visualization 链路，在第五阶段补齐
> 多视图 DTC、候选聚合、运动预测和有界恢复；后续阶段只负责应用入口、评测和训练。

---

## 1. 第一阶段：先定契约

- [x] 完成 `src/instatarget/core/types.py`
  - [x] 定义 `FramePacket`
  - [x] 定义 `DepthSummary`
  - [x] 定义 `MotionState3D`
  - [x] 定义 `TrackResult`
  - [x] 定义 `LocalView`
  - [x] 定义 `SearchPlan`
  - [x] 定义 `TemplateCommand`

- [x] 完成 `src/instatarget/core/protocols.py`
  - [x] 定义 `FrameSource`
  - [x] 定义 `ResultSink`
  - [x] 定义 `TrackerBackend`
  - [x] 定义 `TrackController`

- [x] 完成 `src/instatarget/core/config.py`
  - [x] 统一配置结构
  - [x] 补齐阈值、窗口长度、深度开关

- [x] 完成 `src/instatarget/core/errors.py`
  - [x] 统一异常层级

验收：

- [x] 所有模块都能只依赖 `core` 契约写代码
- [x] `docs/interface.md` 与 `src/instatarget/core` 语义一致

第五阶段开始前必须在保持严格 schema 校验的前提下，向 `tracking`/`recovery` 增加 DTC 参数；
这属于配置契约的向后兼容扩展，不得在 controller 内私自读取未登记字段。

---

## 2. 第二阶段：先把几何跑通

- [x] 完成 `src/instatarget/geometry/spherical_geometry.py`
  - [x] 球面坐标与单位向量转换
  - [x] yaw / pitch 规范化

- [x] 完成 `src/instatarget/geometry/bfov_projector.py`
  - [x] ERP 到 BFoV 的裁剪
  - [x] BFoV 到局部框的回投影

- [x] 完成 `src/instatarget/geometry/projection_math.py`
  - [x] 视场、角度、像素之间的转换

- [x] 完成 `src/instatarget/geometry/seam.py`
  - [x] 跨经线框处理

验收：

- [x] RGB 裁剪结果稳定
- [x] 深度和 RGB 能同步裁剪到同一视场
- [x] 不引入状态机逻辑

---

## 3. 第三阶段：RGB-only 后端（已完成）

- [x] 完成 `src/instatarget/tracker/hit_backend.py`
  - [x] 只保留 HiT 主干推理接口
  - [x] 输出局部框与外观相关分数

- [x] 完成 `src/instatarget/tracker/observation.py`
  - [x] 定义后端输出结构
  - [x] 让 `modelScore / appearanceScore / depthScore / fusedScore` 可落地

- [x] 完成 `src/instatarget/tracker/backend.py`
  - [x] 串起 `initialize / infer / close`
  - [x] 支持 `rgb_only`，并保留 RGB-D 扩展接口

- [x] 完成 `src/instatarget/tracker/template.py`
  - [x] 模板缓存
  - [x] 模板命令执行

验收：

- [x] 不接深度时也能输出稳定 `LocalObservation`
- [x] 后端可以独立被单测调用

---

## 4. 第四阶段：深度链路和融合（已完成）

- [x] 完成 `src/instatarget/tracker/depth_preprocessor.py`
  - [x] 深度归一化
  - [x] 估计局部背景面
  - [x] 浮雕式深度颜色化
  - [x] 轮廓增强
  - [x] 缺失值掩码

- [x] 完成 `src/instatarget/tracker/depth_encoder.py`
  - [x] 深度伪彩色图编码
  - [x] 第二个 HiT 深度分支适配

- [x] 完成 `src/instatarget/tracker/fusion_head.py`
  - [x] 融合 RGB HiT、深度 HiT、模板上下文与轻量几何参数
  - [x] 输出 `fusedScore`

- [x] 回填 `src/instatarget/tracker/backend.py`
  - [x] 保持 `rgb_only` 退化路径
  - [x] 支持 `rgb_depth`
  - [x] 输出 `depthScore` 和 `fusedScore`

验收：

- [x] `rgb_only` 和 `rgb_depth` 都能走通
- [x] 深度颜色化能明显增强轮廓对比
- [x] `fusedScore` 由后端统一产生
- [x] 深度到 RGB 的转换留在 `TrackerBackend`，geometry 只负责 RGB/Depth 同步裁剪
- [x] 深度无效的单视图自动退化为 RGB-only，不影响同一 batch 的其他视图

---

## 5. 第五阶段：控制层 DTC

- [ ] 完成 `src/instatarget/controller/motion_estimator.py`
  - [ ] 常速度 Alpha-Beta 基线；Kalman 作为可替换实现
  - [ ] 在单位向量空间处理 yaw wrap
  - [ ] 深度无效时只更新方向，不污染 range 状态
  - [ ] `MotionState3D` 更新及窗口历史维护

- [ ] 完成 `src/instatarget/controller/decision_gate.py`
  - [ ] `candidateMinScore` 单图候选过滤
  - [ ] 组合 `fusedScore`、运动、尺度和可用深度一致性
  - [ ] 处理 `uncertainThreshold / acceptThreshold / recoverAcceptThreshold`
  - [ ] 连续帧计数和滞回，避免状态抖动

- [ ] 完成 DTC 单帧候选聚合（可先放在 `depth_aware_track_controller.py` 的私有组件）
  - [ ] 每帧固定生成 yaw `-120°/0°/+120°` 的 guard triplet
  - [ ] 生成预测中心、尺度和恢复环视图，去重并遵守 `maxViewsPerFrame >= 3`
  - [ ] 将局部结果回投影为 `ProjectedObservation`
  - [ ] 按球面角距离/尺度聚类，输出单帧预测框和单帧置信度

- [ ] 完成 `src/instatarget/controller/state_machine.py`
  - [ ] `INIT -> TRACKING -> UNCERTAIN -> RECOVERING -> TRACKING`
  - [ ] `LOST` 分支
  - [ ] 找回后从当前帧重建运动窗口，丢弃未来预测假设
  - [ ] 预测输出使用 `valid=false`，不得伪装成观测

- [ ] 完成 `src/instatarget/controller/recovery_planner.py`
  - [ ] 上下文宽高至少为初始/预测框对应尺寸的 2 倍
  - [ ] 扩窗、环搜、全景粗搜的有限预算
  - [ ] 同一帧最多一次 batch，不在单图上无限重试
  - [ ] 跨后续帧维护有限预测假设

- [ ] 完成 `src/instatarget/controller/template_policy.py`
  - [ ] 模板更新条件
  - [ ] 模板保护
  - [ ] `UNCERTAIN / RECOVERING / LOST` 阶段强制 `KEEP`
  - [ ] anchor 永久保留，recent/stable 按连续高置信更新

- [ ] 完成 `src/instatarget/controller/depth_aware_track_controller.py`
  - [ ] 串起初始化、窗口预测、guard/自适应多视图、候选聚合和状态更新
  - [ ] 只消费后端分数和深度摘要，不重做后端融合
  - [ ] 校验 `sequenceId / frameIndex / stateRevision`

验收：

- [ ] 每帧至少三张 guard 视图且不超过视图预算
- [ ] 正常跟踪、短时遮挡恢复和 LOST 降频搜索都能切换
- [ ] 单图低分不污染单帧聚合，单帧结果来自多视图一致性
- [ ] 控制层不直接做深度编码或 MLP 融合
- [ ] RGB-only 与 RGB-D 的 DTC 行为均可运行

---

## 6. 第六阶段：应用入口与 I/O

- [ ] 完成 `src/instatarget/io/video_source.py`
  - [ ] 视频 / 序列读取

- [ ] 完成 `src/instatarget/io/result_sink.py`
  - [ ] 结果写出

- [ ] 完成 `src/instatarget/io/result_writer.py`
  - [ ] 标准输出格式落盘

- [ ] 完成 `src/instatarget/app/track.py`
  - [ ] 普通视频主入口

- [ ] 完成 `src/instatarget/app/track_airsim360.py`
  - [ ] AirSim360 入口

- [ ] 完成 `src/instatarget/app/driver.py`
  - [ ] 将 `FrameSource -> DTC -> TrackerBackend -> Sink` 串起来

验收：

- [ ] 能从命令行跑一条完整序列
- [ ] 有正确的结果文件输出

---

## 7. 第七阶段：适配器和评测

- [ ] 完成 `src/instatarget/adapters/competition_adapter.py`
  - [ ] 官方格式转换
  - [ ] 循环框或双框兼容策略

- [ ] 完成 `src/instatarget/eval/spherical_metrics.py`
  - [ ] 球面角度指标

- [ ] 完成 `src/instatarget/eval/otb_metrics.py`
  - [ ] 常规跟踪指标

- [ ] 完成 `src/instatarget/eval/profiler.py`
  - [ ] 性能统计

验收：

- [ ] 官方结果格式能稳定导出
- [ ] 评测脚本能读取结果并统计指标

---

## 8. 第八阶段：训练链路

- [ ] 完成 `src/instatarget/training/dataset.py`
  - [ ] 训练样本读取
  - [ ] `rgb_only / rgb_depth` 区分

- [ ] 完成 `src/instatarget/training/losses.py`
  - [ ] 分类损失
  - [ ] 回归损失
  - [ ] IoU / 融合一致性损失

- [ ] 完成 `src/instatarget/training/train_backend.py`
  - [ ] 冻结主干
  - [ ] 只训练深度模块和融合头

验收：

- [ ] 训练和推理接口共用同一套数据契约
- [ ] 训练结果能回灌到后端

---

## 推荐执行顺序

1. `core`
2. `geometry`
3. `tracker` 的 `rgb_only`
4. `depth_preprocessor + depth_encoder + fusion_head`
5. `controller`
6. `app / io`
7. `adapters / eval`
8. `training`

这条顺序的核心原则是：先让系统能跑，再让系统会判断，最后让系统会变聪明。
