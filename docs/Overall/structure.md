# 系统设计

系统将 ERP 全景帧投影为局部透视视图，由 ARTrackV2-B-256 预测局部框，
再通过球面 Geometry 回投并由 Controller 完成候选融合、运动预测和结果提交。

模型边界位于 `tracker/artrack_model.py`；Controller、Geometry、I/O 和输出协议
不依赖第三方模型类型，因此可独立验证 ARTrackV2 的局部 IoU 与端到端 BFoV IoU。
