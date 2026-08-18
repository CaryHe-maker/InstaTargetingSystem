# 验证流程

## 算法修改前

固定数据序列、instance ID、配置、模型权重和随机/确定性设置。保存基线 tracking.txt、time.json、指标摘要以及至少一段 local/backend/geometry/final 可视化。

## 分层检查

1. Core/配置测试保证协议和参数合法。
2. Geometry 测试保证一次边界回投、紧致 BFoV、直接 ERP bbox 和跨缝语义。
3. Tracker 测试保证 batch 输入输出顺序、返回数量、真实批量 session、无 `inferBatch` session 的串行回退，以及 RGB-D 深度缺失回退。
4. Controller 测试保证协方差运动评分、70/30 SingleScore、正常线程只在 TRACKING/UNCERTAIN 间转移且均使用 4+4、低于 LT 的 HARD_MISS 保持 UNCERTAIN，以及保留的显式 LOST 组件仍可执行 6+4 单轮与重捕获；同时覆盖第一轮 Fusor 搜索中心、最终跨轮融合和参考面积裁剪的三个分支。
5. Driver 集成测试保证计时和生命周期。
6. AirSim360/Competition 测试保证实际格式和帧数。

## 指标组合

不能只报告 meanIoU。至少同时报告循环 IoU/AUC、BFoV spherical IoU、球面中心误差、宽高误差、`envelopeInflation`、每状态帧数、平均视图数、每轮 batch size、每帧模型 forward 数、可靠融合比例、valid 比例和 P95 处理时间。分数改动还应报告两个输入的校准指标和候选排序 AUC。

## 回归命令

```powershell
& ".venv\Scripts\python.exe" -m pytest -p no:cacheprovider tests
& ".venv\Scripts\python.exe" -m ruff check src tests tools
git diff --check
```

若全仓静态检查存在历史问题，必须至少对本次修改文件定向检查，并在交付说明中区分历史问题与新增问题。

