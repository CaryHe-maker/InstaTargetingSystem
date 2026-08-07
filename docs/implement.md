# InstaTargetingSystem 实现清单

> 目标：按最小可运行链路推进，先跑通 `rgb_only`，再补 `depth`、融合和恢复。

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

---

## 2. 第二阶段：先把几何跑通

- [ ] 完成 `src/instatarget/geometry/spherical_geometry.py`
  - [ ] 球面坐标与单位向量转换
  - [ ] yaw / pitch 规范化

- [ ] 完成 `src/instatarget/geometry/bfov_projector.py`
  - [ ] ERP 到 BFoV 的裁剪
  - [ ] BFoV 到局部框的回投影

- [ ] 完成 `src/instatarget/geometry/projection_math.py`
  - [ ] 视场、角度、像素之间的转换

- [ ] 完成 `src/instatarget/geometry/seam.py`
  - [ ] 跨经线框处理

验收：

- [ ] RGB 裁剪结果稳定
- [ ] 深度和 RGB 能同步裁剪到同一视场
- [ ] 不引入状态机逻辑

---

## 3. 第三阶段：先做 RGB-only 后端

- [ ] 完成 `src/instatarget/tracker/hit_backend.py`
  - [ ] 只保留 HiT 主干推理接口
  - [ ] 输出局部框与外观相关分数

- [ ] 完成 `src/instatarget/tracker/observation.py`
  - [ ] 定义后端输出结构
  - [ ] 让 `modelScore / appearanceScore / depthScore / fusedScore` 可落地

- [ ] 完成 `src/instatarget/tracker/backend.py`
  - [ ] 串起 `initialize / infer / close`
  - [ ] 先支持 `rgb_only`

- [ ] 完成 `src/instatarget/tracker/template.py`
  - [ ] 模板缓存
  - [ ] 模板命令执行

验收：

- [ ] 不接深度时也能输出稳定 `LocalObservation`
- [ ] 后端可以独立被单测调用

---

## 4. 第四阶段：补深度链路和融合

- [ ] 完成 `src/instatarget/tracker/depth_preprocessor.py`
  - [ ] 深度归一化
  - [ ] 裁剪
  - [ ] 缺失值掩码
  - [ ] 生成 `DepthSummary`

- [ ] 完成 `src/instatarget/tracker/depth_encoder.py`
  - [ ] 深度特征编码

- [ ] 完成 `src/instatarget/tracker/fusion_head.py`
  - [ ] 融合 RGB 特征、深度特征、模板上下文
  - [ ] 输出 `fusedScore`

- [ ] 回填 `src/instatarget/tracker/backend.py`
  - [ ] 支持 `rgb_depth`
  - [ ] 输出 `depthScore` 和 `fusedScore`

验收：

- [ ] `rgb_only` 和 `rgb_depth` 都能走通
- [ ] `fusedScore` 由后端统一产生

---

## 5. 第五阶段：控制层 DTC

- [ ] 完成 `src/instatarget/controller/motion_estimator.py`
  - [ ] 常速度 / Kalman 预测
  - [ ] `MotionState3D` 更新

- [ ] 完成 `src/instatarget/controller/decision_gate.py`
  - [ ] 接收阈值
  - [ ] 处理 `acceptThreshold / uncertainThreshold`

- [ ] 完成 `src/instatarget/controller/state_machine.py`
  - [ ] `INIT -> TRACKING -> UNCERTAIN -> RECOVERING -> TRACKING`
  - [ ] `LOST` 分支

- [ ] 完成 `src/instatarget/controller/recovery_planner.py`
  - [ ] 扩窗
  - [ ] 环搜
  - [ ] 全景粗搜

- [ ] 完成 `src/instatarget/controller/template_policy.py`
  - [ ] 模板更新条件
  - [ ] 模板保护

- [ ] 完成 `src/instatarget/controller/depth_aware_track_controller.py`
  - [ ] 串起窗口预测、候选选择、状态更新
  - [ ] 只消费后端分数，不重做后端融合

验收：

- [ ] 正常跟踪和恢复模式都能切换
- [ ] 控制层不再直接做深度编码或 MLP 融合

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
