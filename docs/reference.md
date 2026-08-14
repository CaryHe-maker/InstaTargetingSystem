# 参考与依赖

## 直接运行依赖

- PyTorch `2.6.x` 与 torchvision `0.21.x`：加载和执行 HiT-Small。
- `third_party/HiT`：官方 HiT-Small 源码树，运行时通过 `HIT_ROOT` 或仓库相对路径导入。
- `models/hit_small.pth`：HiT-Small 权重文件，必须包含 `net` 状态字典。
- OpenCV：视频解码、图像缩放、边界填充和 PNG 处理。
- NumPy、PyYAML、timm、EasyDict、tensorboardX：数组、配置和官方 HiT 依赖。

## 项目内部实现

球面投影、ERP 接缝处理、运动估计、状态机、恢复视图规划、深度预处理和指标计算均位于 `src/instatarget`。这些模块通过 `core` 数据契约连接，不要求调用方了解第三方模型内部结构。

## 背景资料

HiT、全景视频目标跟踪、球面投影和 AirSim360 数据格式属于项目设计所参考的公开技术背景。背景资料用于说明算法语境，不代表仓库额外加载或调用相应论文中的其他模型、训练脚本或推理服务。
