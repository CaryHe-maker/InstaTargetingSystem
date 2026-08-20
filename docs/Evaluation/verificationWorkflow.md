# 验证流程

## 算法修改前

固定数据序列、instance ID、配置、模型权重和随机/确定性设置。保存基线 tracking.txt、time.json、指标摘要以及至少一段 local/backend/geometry/final 可视化。

## 分层检查

1. Core/配置测试保证协议和参数合法。
2. Geometry 测试保证一次边界回投、紧致 BFoV、直接 ERP bbox 和跨缝语义。
3. Tracker 测试保证 RGB batch 输入输出顺序、返回数量、真实批量 session，以及无 `inferBatch` session 的串行回退。
4. Controller 测试保证 Stage 3 产物驱动的 50/50 SingleScore、正常线程只在 TRACKING/UNCERTAIN 间转移且均使用 4+4、低于 LT 的 HARD_MISS 保持 UNCERTAIN，以及保留的显式 LOST 组件仍可执行 6+4 单轮；同时覆盖第一轮 Fusor 搜索中心、最终跨轮融合和 `best_source` 几何。
5. Driver 集成测试保证计时和生命周期。
6. AirSim360/Competition 测试保证实际格式和帧数。

## 指标组合

不能只报告 meanIoU。至少同时报告循环 IoU/AUC、tracking loss rate 与 lost frame count、BFoV spherical IoU、球面中心误差、宽高误差、`envelopeInflation`、每状态帧数、平均视图数、每轮 batch size、每帧模型 forward 数、可靠融合比例、valid 比例和 P50/P95/P99 处理时间。分数改动还应报告校准指标和候选排序 AUC。

真实评估命令必须使用 `E:\NewDownload\train\manifest.jsonl`，评估工具会拒绝位于该 canonical root 之外的 manifest 或视频。最终 holdout 在模型、校准、Controller 与流水线全部冻结前禁止读取。

性能/精度 A/B 使用 `tools/compare_evaluation_ab.py`。工具按 frame/round/view key 对齐候选，比较 model/presence/quality/appearance/motion/SingleScore、局部框、投影框/BFoV 和最终 TrackResult，同时拒绝任何非有限数。普通低风险优化要求零差异；`--fp16-gates` 允许数值变化，但要求 mean IoU、success@0.5、loss rate、absent FPR 和 P95 同时过门槛。

发布前运行 `tools/verify_release_artifacts.py`。它核对 checkpoint/calibration SHA-256、RGB-only YAML、Docker build context、Git 跟踪文件和严格 7 层结构；传入 `--image-tar` 时还检查导出镜像的实际 layer 数。GitHub Actions 在全新 clone 上执行同一脚本。

## 回归命令

```powershell
& ".venv\Scripts\python.exe" -m pytest -p no:cacheprovider tests
& ".venv\Scripts\python.exe" -m ruff check src tests tools
& ".venv\Scripts\python.exe" tools\verify_release_artifacts.py
git diff --check
```

若全仓静态检查存在历史问题，必须至少对本次修改文件定向检查，并在交付说明中区分历史问题与新增问题。

