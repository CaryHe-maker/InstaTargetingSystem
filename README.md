# InstaTargetingSystem

360 全景单目标跟踪系统。项目已完成几何裁剪、RGB-only / RGB-D 后端、控制层、结果输出、评估和可视化，训练链路保留在 `src/instatarget/training/` 下作为后续扩展入口。

## 目录
- `src/instatarget/core`：数据类型、协议、配置和错误
- `src/instatarget/geometry`：ERP 与 BFoV 的裁剪、回投影和跨经线处理
- `src/instatarget/tracker`：HiT 主干、深度分支、融合头和局部观测
- `src/instatarget/controller`：多视图规划、状态机、恢复和模板更新
- `src/instatarget/io`：帧读取、结果写入和 AirSim360 数据接入
- `src/instatarget/visualization`：中间结果 PNG 记录
- `src/instatarget/eval`：OTB 风格指标和结果读取
- `src/instatarget/training`：后续训练入口

## 运行
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

For the folder-based `raw/depth/semantic/instance` smoke test, use
`tools/run_airsim360_dataset.py`; it
writes tracking output plus modality and stage visualizations. See
[`docs/Airsim360DataSolution.md`](docs/Airsim360DataSolution.md).

## 数据
- AirSim360 序列默认包含 `rgb/`、`depth/`、`semantic/`、`instance/` 和 `meta.json`
- `data/airsim360/nyc_sample/` 是规范化的本地样例，可用于检查读入、可视化和伪标注生成
- 不指定 `--output-dir` 时，样例结果自动写入 `artifacts/airsim360/nyc_sample/output_N/`
- 结果文件采用逐行 `xPx,yPx,widthPx,heightPx` 文本格式

## 输出
- 开发期结果文件会先写入 `.partial`
- `FileResultSink.finalize()` 成功后才会原子替换为最终输出
- `CompetitionAdapter` 负责比赛格式适配

## 配置
- `configs/RGBonly.yaml`：RGB-only
- `configs/RGBD.yaml`：RGB-D
- `docs/hyperparameters.md`：已登记的超参数和约束

## 文档
- [架构](docs/design.md)
- [接口](docs/interface.md)
- [实现说明](docs/implement.md)
- [数据规范](docs/data.md)
- [可视化](docs/modules/visualization.md)
- [训练规约](docs/ModelTraning.md)

## 当前状态
- 运行链路、评估链路和可视化链路已落地
- 训练链路尚未接入真实训练器，当前只保留规范和入口
