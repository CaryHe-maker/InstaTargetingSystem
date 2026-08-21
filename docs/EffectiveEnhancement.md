# feat/Effeciency 实现说明

## 1. 分支与基线

本分支是 `feat/Effeciency`，从 `main` 的 `PostTrainingV1.4` 提交
`449cb8d` 创建。它不是实验变体集合，而是将 Pipeline + GPU Geometry 固化为默认运行路径的生产模板。

`PostTrainingV1.4` 相对 `PostTraining V1`（`d450e15`）的能力已保留。特别复核了以下存在重叠修改的文件：

- `src/instatarget/app/driver.py`：保留 V1.4 的安全球面投影过滤 `_projectValidObservations`。
- `src/instatarget/tracker/pytorch_hit_session.py`：保留 V1.4 的统一 `_constructRuntimeModel`、CPU checkpoint 校验 `validateHiTCheckpoint`、安全 checkpoint 加载和 vendor runtime 构建；在其上增加 device batch、显存统计和 CUDA 优化。
- V1.4 其余配置、评分、模型加载、提交校验和文档基线均从 `main` 继承，没有回退到 V1。

## 2. 默认运行架构

正常调用 `buildRuntime(config)` 即使用以下链路：

```text
ERP Frame
   │
   ├─ 一次 H2D：uint8 ERP → CUDA FP32 frame tensor
   │
   ├─ GPU Geometry：CUDA perspective vectors + seam-aware bilinear sampling
   │                    + normalization
   │
   ├─ HiT device batch：直接消费 [C,H,W] CUDA tensor
   │
   ├─ Round 1：候选与 provisional prediction
   │
   ├─ Round 2：使用 provisional prediction 重新生成运动规划
   │
   └─ Commit：最终测量替换 provisional，正式运动历史只写入一次
```

帧解码由 `_PrefetchReader` 在后台线程执行，队列保持输入顺序；推理线程只消费已解码帧。每帧处理结束后显式释放 GPU frame tensor，避免跨帧 CUDA 引用导致显存增长。

## 3. GPU Geometry 实现

文件：`src/instatarget/geometry/gpu_geometry.py`

- ERP 帧只执行一次 Host→Device 传输。
- 透视网格、yaw/pitch/roll 变换、球面投影和双线性采样均在 CUDA 上完成。
- seam 方向使用循环 x 坐标，极点方向对 y 坐标进行边界裁剪。
- 输出为标准化 FP32 `cuda:0` tensor，兼容 HiT 的 `[3, H, W]` 输入。
- 跨帧只保留局部 CPU 模板副本，不保留 CUDA `deviceRgb`。

证据字段：`imageRoundTrips=0`、`gridCpuTransfers=0`、`frameTensorDevice=cuda:0`、`frameTensorDtype=torch.float32`。

## 4. Pipeline 预测语义

Pipeline 不只是 decode prefetch。`TrackControllerImpl` 在同一帧事务中：

1. Round 1 完成候选评估后生成 provisional prediction。
2. provisional prediction 不修改正式 motion history。
3. Round 2 读取 provisional prediction，更新中心、尺度和后续 view planning。
4. 最终 commit 时使用正式测量替换 provisional，并只提交一次运动历史。

控制器记录 `round1PredictionRevision`、`round2UsesProvisionalPrediction`、`finalStateRevision` 和 `provisionalReplacedAtCommit`，用于运行时审计。

## 5. `seq_0045` 验证结果

测试序列：`train_sim/seq_0045`，1296 帧，FP32，8 views / 2 forwards，四组使用同一 checkpoint、config 和 dataset。

| 方案 | IoU | Spherical IoU | Success@0.5 | Absent FPR | P50 | P95 | P99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 对照组 | 0.1763 | 0.1696 | 0.1462 | 0.5607 | 355.9 ms | 501.1 ms | 628.9 ms |
| GPU Geometry only | 0.2138 | 0.2079 | 0.1765 | **0.2428** | 90.0 ms | 118.4 ms | 135.4 ms |
| Pipeline only | **0.2405** | **0.2341** | **0.1979** | 0.5318 | 280.6 ms | 325.0 ms | 382.8 ms |
| Pipeline + GPU Geometry | 0.2204 | 0.2148 | 0.1836 | 0.5434 | **86.9 ms** | **115.4 ms** | **123.9 ms** |

组合方案相对对照组的 P50 提升约 75.6%，P99 提升约 80.3%。该序列上 `pipeline_only` 精度最高，`gpu_geometry_only` 的 Absent FPR 最低；因此速度最优不等于精度最优，不能仅凭单序列结果更换所有场景的策略。

## 6. V1.4 保留审计

使用 `git diff d450e15..origin/main` 与 `git diff origin/main..HEAD` 做交集复核：

- `feat/Effeciency` 的祖先包含 `origin/main`，即完整包含 `PostTrainingV1.4`。
- 重叠文件仅为 `driver.py` 和 `pytorch_hit_session.py`；两者已分别恢复 V1.4 的投影容错和统一 HiT 模型构建接口。
- V1.4 的安全 checkpoint、评分校准、配置校验和提交协议均通过测试。

## 7. 验证与运行要求

- 单元测试和驱动集成测试：`138 passed`。
- CUDA Geometry 像素回归：P99 像素误差为 0，最大误差为 1 个 uint8 灰度级。
- `git diff --check` 通过，源码可编译。
- 生产默认路径要求可用 NVIDIA CUDA；无 CUDA 时 `GpuGeometryImpl` 会明确抛出 `GeometryError`，不会静默退回 CPU 假路径。

本分支已移除实验运行器 `tools/run_trying_plan.py` 以及 IoU/FOV/ERP crop 等变体入口；这些内容只保留在实验报告文档中，不参与生产运行。
