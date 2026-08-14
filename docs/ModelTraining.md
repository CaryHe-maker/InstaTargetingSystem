# HiT Model Training

当前 RGB-only 与 RGBD 共享官方 `HiT_Small` 网络结构，但保留两份独立 YAML，便于后续使用不同权重、冻结层数和训练数据。

## Runtime Baseline

- Official source: `third_party/HiT`
- Official checkpoint: `models/hit_small.pth`
- Device: CUDA GPU
- Runtime precision: FP32
- RGB-only input: geometry 生成的原始局部 RGB
- RGBD input: 深度图预测边缘后，只在边缘像素改色得到的增强 RGB
- Output score: `fusedScore = appearanceScore`
- RGBD 没有第二个 HiT、独立深度编码器或融合头

官方 HiT_Small checkpoint 的 corner head 在 FP16 下可能产生非有限框，因此当前两份运行配置都使用 CUDA FP32。代码保留 CUDA autocast 支持，但在启用 FP16 前必须针对目标权重和数据完成稳定性验证。

## Separate Configurations

`configs/RGBonly.yaml` 和 `configs/RGBD.yaml` 必须继续独立维护。后续训练可以为两种模式设置不同的 `model.weights`，但模型接口和输出字段保持一致。

建议训练流程：

1. 从官方 HiT_Small 权重初始化 RGB-only 基线。
2. 使用原始 RGB 局部图训练或微调 RGB-only 权重。
3. 使用深度边缘增强 RGB 训练 RGBD 权重。
4. RGBD 初期冻结 backbone 前几层，只训练后段 backbone、neck 和 box head。
5. 验证收敛后逐步解冻，避免边缘改色破坏官方预训练特征。
6. 分别在两份 YAML 中登记对应权重，不增加融合头。

训练或导出后的模型必须满足：

- 输入为 `uint8 [H,W,3]` RGB，经 ImageNet mean/std 归一化；
- RGBD 非边缘像素必须与原 RGB 逐像素一致；
- `depthScore` 固定为 `0.0`；
- `fusedScore` 等于单 HiT 的 `appearanceScore`；
- 模板初始化、搜索帧和在线模板更新使用同一种输入增强方式。

## Environment

完整依赖、版本、源码和权重下载方式见 `docs/environment.md`。训练环境至少需要 CUDA PyTorch、torchvision、timm 0.5.4、yacs、easydict、scipy、pandas 和 tensorboard。

## Short Runtime Commands

安装项目后可直接运行：

```powershell
run -RGB_only /data/airsim360/nyc_sample /artifacts/airsim360/nyc_sample/test_rgb 2497023
run -RGBD /data/airsim360/nyc_sample /artifacts/airsim360/nyc_sample/test_rgbd 2497023
```

可通过 `--config` 指定训练后配置，通过 `--max-frames` 做短序列验证。
