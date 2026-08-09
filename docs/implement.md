# InstaTargetingSystem 交付状态

> 这是本仓库当前的最终交付说明。系统已完成球面几何、RGB-only/RGB-D 跟踪后端、控制层、应用入口、I/O、竞赛适配和评测链路；训练链路仍保留为后续扩展。

---

## 已交付

- `core`：统一数据类型、协议、配置和错误层。
- `geometry`：ERP 与 BFoV 之间的裁剪、回投影和跨经线处理。
- `tracker`：HiT 主干、深度预处理、双分支融合和模板命令执行。
- `controller`：DTC 负责多视图计划、候选聚合、运动预测、状态机和恢复策略。
- `app / io`：命令行入口、视频/序列读取、结果写出和 AirSim360 数据接入。
- `adapters / eval`：官方结果格式转换、球面指标、OTB 指标和性能统计。
- `visualization`：局部 RGB、深度诊断图、后端框和回投影框的无损记录。

## 运行入口

```bash
python -m instatarget.track \
  --input input.mp4 \
  --init-box 120.0,80.0,64.0,96.0 \
  --output result.txt \
  --config configs/RGBonly.yaml
```

```bash
python -m instatarget.track_airsim360 \
  --dataset-root data/AirSim360 \
  --sequence NYC_001 \
  --target-instance 305 \
  --output result.txt \
  --config configs/RGBonly.yaml
```

## 输出

- 开发期结果文件采用每行一个框的纯文本格式：`xPx,yPx,widthPx,heightPx`
- 比赛格式由 `CompetitionAdapter` 统一转换
- 结果文件采用 `.partial` 原子落盘，只有 `finalize()` 成功后才会变成最终输出

## 当前约束

- RGB-only 与 RGB-D 共用同一套控制层和结果协议
- 深度缺失时自动退化为 RGB-only
- 训练链路尚未落地，当前仓库不包含端到端再训练实现

## 最终边界

本项目目前的产品边界是“可运行、可评测、可导出、可诊断”的全景单目标跟踪系统。
后续如果补训练链路，只需要在现有 `training` 目录下扩展，不需要改动当前推理契约。
