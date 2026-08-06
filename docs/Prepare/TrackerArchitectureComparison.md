# LightTrack、HiT/DyHiT、360VOT 对比文档

## 1. 总体结论

本项目最推荐的路线是：

**360VOT 作为全景几何与评测框架，HiT/DyHiT 作为主力跟踪核心，LightTrack 作为轻量 baseline。**

三者定位不同：

1. **LightTrack**：轻量 Siamese/NAS tracker，适合快速建立 baseline。
2. **HiT/DyHiT**：更现代的轻量 Transformer tracker，适合作为主线模型。
3. **360VOT**：全景数据集、表示、几何工具和评测框架，不是单独 tracker。

## 2. 横向对比

| 项目 | 类型 | 主要价值 | 是否直接适合全景 | 推荐角色 |
| --- | --- | --- | --- | --- |
| LightTrack | 轻量单目标 tracker | 快、研究代码完整、适合 baseline | 否 | 基线模型 |
| HiT | 轻量 Transformer tracker | 速度/精度更强，支持 ONNX | 否 | 主力 tracker |
| DyHiT | 动态轻量 tracker/加速框架 | 可调速度和精度，适合复杂度变化 | 否 | 决赛优化路线 |
| 360VOT | 全景 benchmark/toolkit | BFoV、rBFoV、球面评测、全景裁剪 | 是 | 全景适配层 |

## 3. 与本项目需求的匹配度

### 3.1 投影畸变

- LightTrack：不原生处理。
- HiT/DyHiT：不原生处理，但特征表达更强。
- 360VOT：提供 BFoV 裁剪和全景处理参考。

结论：畸变问题主要靠 360VOT 风格几何层解决。

### 3.2 经线跨越

- LightTrack：普通 bbox 跟踪会跳变。
- HiT/DyHiT：普通 bbox 跟踪同样会跳变。
- 360VOT：提供适合球面跟踪的表示参考。

结论：必须在内部使用 yaw/pitch 或 BFoV 表示。

### 3.3 长时跟踪和找回

- LightTrack：需要额外做恢复。
- HiT/DyHiT：需要额外做恢复，但更适合处理困难帧。
- 360VOT：可以帮助分析 CB、LD、SA 等失败属性。

结论：恢复模块需要自研，360VOT 用来评估恢复效果。

### 3.4 实时性

- LightTrack：轻量，容易部署。
- HiT/DyHiT：速度更有竞争力，DyHiT 可动态调节。
- 360VOT：本身不是速度优化模型。

结论：实时主线优先 HiT-Small/HiT-Tiny/DyHiT。

## 4. 推荐架构

```text
全景视频帧
  -> 360VOT 风格 BFoV 裁剪
  -> HiT/DyHiT 局部跟踪
  -> 局部 bbox 回投影
  -> 球面状态更新
  -> 低置信恢复搜索
  -> 输出比赛 bbox
```

## 5. 技术选型

### 5.1 主语言

- **Python**：第一阶段主开发语言。
- **C++**：后期部署优化语言。
- **Shell**：脚本和 Docker 入口。

### 5.2 核心框架

- **PyTorch**：训练和调试。
- **OpenCV**：视频处理和图像变换。
- **NumPy**：坐标和球面几何。
- **ONNX Runtime**：推理部署。
- **TensorRT（可选）**：GPU 加速。
- **CMake**：C++/混合工程管理。
- **Docker**：比赛提交。

## 6. 分阶段决策

### 阶段 1：最小可行版本

选择：

- LightTrack 或 HiT-Small。
- 简单 BFoV 裁剪。
- 普通 bbox 输出。

目标：

- 尽快跑通从视频到逐帧框输出。

### 阶段 2：推荐主线版本

选择：

- HiT-Small 或 HiT-Tiny。
- 360VOT 风格 BFoV 表示。
- 自研跨经线处理。
- 基础丢失恢复。

目标：

- 提升全景场景稳定性。

### 阶段 3：决赛优化版本

选择：

- DyHiT。
- ONNX Runtime 或 TensorRT。
- 动态搜索窗口。
- 多候选恢复。

目标：

- 同时优化 AUC、Success Rate 和 FPS。

## 7. 不推荐路线

### 7.1 只用 LightTrack

风险：

- 全景畸变、跨边界和极点问题都没解决。
- 很可能在普通视频上可用，在全景测试上不稳定。

### 7.2 只用 360VOT

风险：

- 360VOT 不是 tracker。
- 需要外部模型提供逐帧跟踪能力。

### 7.3 直接整帧全景输入 Transformer

风险：

- 分辨率高，速度压力大。
- 极区畸变仍然存在。
- 不利于端侧部署。

## 8. 最终推荐

如果只选一条主线：

**HiT-Small + 360VOT BFoV 几何层 + 自研恢复模块。**

如果同时准备备用方案：

1. LightTrack：保底 baseline。
2. HiT-Small：主力实时方案。
3. HiT-Base：高精度对照。
4. DyHiT：速度/精度动态优化方案。
5. 360VOT：全景评测和几何适配工具链。

## 9. 下一步建议

1. 先用 360VOT 的 BFoV 思路实现本项目几何模块。
2. 然后接 HiT-Small，形成可跑通全景序列的第一版。
3. 再把 LightTrack 作为 baseline，对比速度和精度。
4. 最后尝试 DyHiT 和 ONNX/TensorRT 优化。

