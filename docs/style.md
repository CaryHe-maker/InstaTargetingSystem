# InstaTargetingSystem 工程规范

> 本文档约定 Python、配置、文档、测试和 Git 规范，面向所有贡献者。
> 正确性与可复现性优先于个人偏好。

---

## 1. 命名约定

### 1.1 Python

| 类别 | 风格 | 示例 |
|------|------|------|
| class / enum / protocol | PascalCase | `TrackController`, `TrackStatus` |
| 函数 / 方法 | camelCase | `cropViews()`, `bboxToBfov()` |
| 变量 / 参数 | camelCase | `frameIndex`, `localBox` |
| 常量 | UPPER_SNAKE_CASE | `MAX_RECOVERY_VIEWS` |
| 私有成员 | 前缀 `_` | `_backend`, `_stateRevision` |
| 类型别名 | PascalCase | `FrameIndex`, `SequenceId` |

公共 API 的名称以 `interface.md` 为准。禁止自行混用 snake_case 公共接口。

### 1.2 文件与目录

| 类别 | 风格 | 示例 |
|------|------|------|
| Python 文件 | lower_snake.py | `spherical_geometry.py` |
| Python 包 | lower_snake | `instatarget/geometry/` |
| 配置文件 | lower_snake.yaml | `hit_small.yaml` |
| 测试文件 | test_lower_snake.py | `test_seam_projection.py` |
| Markdown | lower_snake.md | `design.md`, `interface.md` |

### 1.3 单位与坐标

- 像素后缀 `Px`：`widthPx`。
- 弧度后缀 `Rad`：`yawRad`。
- 角度后缀 `Deg`：仅配置层使用，如 `minFovDeg`。
- 时间后缀 `Ns` / `Ms`：`latencyNs`。
- 禁止 `x1/y1/x2/y2` 与 `xywh` 混传；类型名必须注明格式。
- 禁止使用无单位的 `angle`、`timeout`、`size`。

---

## 2. Python 格式

### 2.1 基本格式

- Python 3.11。
- 4 个空格缩进，不用 Tab。
- 行长上限 100 字符。
- UTF-8、LF 行尾、文件末尾一个换行。
- 使用 Ruff 格式化与静态检查，禁止手工对齐制造无意义 diff。
- 所有公共函数和方法必须有完整类型标注。

```python
def buildSearchPlan(
    state: TrackState,
    frame: FramePacket,
    config: RecoveryConfig,
) -> SearchPlan:
    ...
```

### 2.2 Import 顺序

按以下分组，组间一个空行：

1. 标准库。
2. 第三方库。
3. 项目内绝对导入。

```python
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from instatarget.core.types import BFoV
from instatarget.geometry.projector import BfovProjector
```

禁止 `from module import *`；生产代码禁止修改 `sys.path`。

### 2.3 数据类与可变性

- 跨模块消息使用 `@dataclass(frozen=True, slots=True)`。
- 配置加载后不可变。
- 大数组只读传递，修改前显式复制。
- 函数参数禁止使用可变默认值。
- 状态只能由声明的所有者线程修改。

```python
@dataclass(frozen=True, slots=True)
class ViewSpec:
    viewId: int
    bfov: BFoV
    outputWidthPx: int
    outputHeightPx: int
```

### 2.4 类型

- 用 `X | None`，不用 `Optional[X]`。
- 用 `collections.abc` 中的抽象容器描述只读输入。
- NumPy/Tensor 字段必须在注释或类型中说明 shape、dtype、颜色顺序和 device。
- 公共接口禁止无约束的 `dict[str, object]`。
- 禁止用 `Any` 绕过模块协议；第三方边界除外。

---

## 3. 数值与张量规范

### 3.1 图像

| 阶段 | 格式 |
|------|------|
| 解码输出 | RGB `uint8`, HWC, CPU |
| 模型输入 | RGB `float32/float16`, NCHW, device |
| 几何网格 | 归一化坐标 `float32` |
| 模型输出 | `float32` 后处理 |

颜色转换只允许在 I/O 或模型预处理边界执行。变量名需包含 `rgb` 或 `bgr`，禁止使用
含义不明的 `image` 跨模块传递。

### 3.2 角度

- 内部统一弧度。
- `yaw` 使用半开区间 `[-π, π)`。
- 比较 yaw 差必须使用循环角差或单位向量，不得直接相减。
- 近极点运动使用单位向量/切平面，不依赖 yaw。
- 三角函数输入和输出必须在局部变量名中标明单位。

### 3.3 浮点比较

- 禁止直接用 `==` 比较投影结果。
- 测试使用显式绝对/相对容差。
- 阈值写入配置并说明量纲。
- 非有限数必须在模块边界立即拒绝。

---

## 4. 模块设计

### 4.1 依赖方向

```text
core <- geometry <- tracker <- controller <- app
                     ^              |
                     +--------------+
```

- `core` 不依赖 OpenCV、PyTorch 或具体模型。
- `geometry` 不依赖 tracker。
- `tracker` 不解释比赛文件格式。
- `controller` 通过 `TrackerBackend` 协议使用模型，并独占最近 `n` 帧窗口、状态机和最终门控。
- `tracker` 负责深度处理、局部跟踪和 MLP 融合，返回局部观测及融合分数。
- `app` 是模块首次汇合点。
- `eval` 可读取结果，不反向影响跟踪决策。

禁止循环导入。为解决循环导入而移动 import 到函数内部必须附设计说明。

### 4.2 函数与类

- 一个函数只做一项工作；建议不超过 40 行。
- 一个模块只暴露一组相关职责。
- 纯计算优先使用函数；有资源或不变式时使用类。
- 公共类通过构造函数注入依赖，不在内部创建具体后端。
- 布尔参数超过一个时改用配置对象或枚举。

### 4.3 全局状态

禁止可变全局变量。允许：

- 模块级不可变常量。
- 只读类型别名。
- 无状态纯函数。

模型会话、配置、随机数生成器和缓存必须显式传递或由对象拥有。

---

## 5. 注释与文档

- 文档使用中文；源代码注释和 docstring 使用英文。
- 注释解释设计原因、单位、坐标系和非显然约束，不复述代码。
- 公共协议、复杂球面公式和线程所有权必须有 docstring。
- 禁止提交注释掉的代码、过期 TODO 或大段实验日志。
- TODO 格式：`TODO(owner): action and exit condition`。

```python
def wrapYaw(yawRad: float) -> float:
    """Normalize yaw to the half-open interval [-pi, pi)."""
```

Markdown 格式沿用本文档：一个 H1、简短引用说明、水平分隔线、编号 H2、表格与小型
代码块。语言保持短句，术语首次出现时定义。

---

## 6. 错误与日志

### 6.1 错误处理

- 外部输入错误抛项目异常，不用 `assert`。
- `assert` 只用于程序员错误和内部不变式。
- 禁止裸 `except:`，禁止捕获后忽略异常。
- 异常消息包含阶段、对象和期望值，不包含大数组。
- 资源使用上下文管理器或显式 `close()`。

```python
if frame.frameIndex != expectedFrameIndex:
    raise ProtocolError(
        f"frame order mismatch: expected={expectedFrameIndex}, "
        f"actual={frame.frameIndex}"
    )
```

### 6.2 日志

- 使用结构化 logger，不使用 `print()`。
- `stdout` 保留给比赛结果；日志写 `stderr`。
- 每条逐帧日志至少包含 `sequenceId`、`frameIndex`、`status`。
- 默认不记录图像、权重路径中的凭据或环境变量。
- 高频调试日志必须可关闭。

---

## 7. 并发规范

- 每个有状态组件只有一个写线程。
- 跨线程对象不可变；大缓冲区使用明确生命周期的只读句柄。
- 所有消息携带帧号；状态相关消息额外携带 revision。
- 队列必须有容量，不使用无界队列。
- 比赛模式禁止丢帧；满队列使用背压。
- 禁止持锁执行解码、GPU 推理或文件 I/O。
- 停止信号必须唤醒所有阻塞队列。
- 不允许通过增加锁修补不清晰的所有权；先调整所有权设计。

---

## 8. 配置与资源

- 所有超参数进入 YAML，不写死在算法代码中。
- 配置包含 `schemaVersion`，未知字段报错。
- 路径使用 `pathlib.Path`，相对配置文件解析。
- 权重、数据集、推理引擎和结果大文件不提交 Git。
- 权重必须记录来源、许可证、SHA-256 和预处理版本。
- 生产运行禁止自动下载资源。
- 密钥、令牌和用户绝对路径禁止写入配置或日志。

---

## 9. 测试规范

### 9.1 测试层级

| 层级 | 范围 | 要求 |
|------|------|------|
| 单元测试 | 几何、框、状态转移、配置 | 快速、无 GPU、确定性 |
| 集成测试 | 短视频、HiT 后端、I/O | 固定权重和预期结果 |
| 回归测试 | 经线、极点、遮挡、找回 | 每个修复对应固定用例 |
| 后端一致性 | PyTorch/ONNX/TensorRT | 框和分数误差受限 |
| 性能测试 | 端到端与分阶段耗时 | 独立报告，不设脆弱单测阈值 |

### 9.2 几何必测点

- ERP 四角和赤道中心。
- `yaw = -π/+π` 两侧的连续性。
- `pitch` 接近 `±π/2`。
- 跨经线框和普通框。
- BFoV 裁剪后回投影往返误差。
- 多种宽高比、FOV 和分辨率。

### 9.3 测试命名

```python
def testLocalBoxToBfovPreservesCenterAcrossSeam() -> None:
    ...
```

测试名描述行为和条件。一个测试只验证一个失败原因。

---

## 10. 可复现性

- 固定 Python、CUDA、PyTorch、OpenCV 和后端版本。
- 固定 Python、NumPy 和 PyTorch 随机种子。
- 记录 Git commit、配置哈希、权重哈希和设备信息。
- 明确启用确定性模式产生的性能影响。
- 评测时保存逐序列结果，不只保存汇总分数。
- FPS 使用 warm-up 后的单调时钟，并说明是否包含解码。

---

## 11. Docker 规范

- 基础镜像和依赖使用固定版本或 digest。
- 构建与运行阶段分离，最终镜像不含训练数据和缓存。
- 使用非 root 用户运行，除非评测平台明确要求。
- 入口命令使用 exec form，并正确转发退出码和信号。
- 运行时默认无网络，模型权重预置于镜像或挂载目录。
- 镜像构建完成后执行 smoke test 和固定短序列回归。

---

## 12. Git 提交

### 12.1 Commit Message

```text
<type>: <short description>

[optional rationale and validation]
```

| type | 含义 |
|------|------|
| `feat` | 新功能 |
| `fix` | 缺陷修复 |
| `refactor` | 无行为变化的重构 |
| `docs` | 文档 |
| `test` | 测试 |
| `perf` | 性能优化 |
| `chore` | 构建、依赖、工具 |

提交只包含一个逻辑变更。禁止提交权重、数据、生成结果和本地 IDE 配置。

### 12.2 分支

- `main`：可运行主线。
- `feat/<name>`：功能开发。
- `fix/<name>`：缺陷修复。
- 合并前运行格式、类型、单元和相关回归测试。

---

## 13. 快速检查清单

- [ ] 公共 API 与 `interface.md` 一致。
- [ ] 坐标格式、shape、dtype、颜色和单位明确。
- [ ] 无可变全局状态或跨线程隐式共享。
- [ ] 跨经线与近极点逻辑使用球面表示。
- [ ] 异常未被吞掉，比赛输出未混入日志。
- [ ] 新阈值已进入配置并通过校验。
- [ ] 新行为有单元或回归测试。
- [ ] 依赖、权重和运行信息可复现。
- [ ] Ruff、类型检查和测试通过。
- [ ] `git diff --check` 无空白错误。
