# 验证流程

## 算法修改前

固定数据序列、instance ID、配置、模型权重和随机/确定性设置。保存基线 tracking.txt、time.json、指标摘要以及至少一段 local/backend/geometry/final 可视化。

## 分层检查

1. Core/配置测试保证协议和参数合法。
2. Geometry 测试保证投影和跨缝不变。
3. Tracker 测试保证局部分数与 RGB-D 回退。
4. Controller 测试保证分轮、融合和状态转移。
5. Driver 集成测试保证计时和生命周期。
6. AirSim360/Competition 测试保证实际格式和帧数。

## 指标组合

不能只报告 meanIoU。至少同时报告循环 IoU/AUC、球面中心误差、每状态帧数、平均视图数、可靠融合比例、valid 比例和处理时间。分数改动还应报告校准指标。

## 回归命令

```powershell
& ".venv\Scripts\python.exe" -m pytest -p no:cacheprovider tests
& ".venv\Scripts\python.exe" -m ruff check src tests tools
git diff --check
```

若全仓静态检查存在历史问题，必须至少对本次修改文件定向检查，并在交付说明中区分历史问题与新增问题。

