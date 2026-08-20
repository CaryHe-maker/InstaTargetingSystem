# PostTrainingV1 之后变更与 AI 合并指南

> 文档版本：PostTrainingV1.3  
> 适用分支：main  
> 基线提交：d450e15（PostTraining V1）  
> 当前代码提交：227ca0f（Fix wrapped BFoV projection）

## 1. 文档目的

本文是从 PostTrainingV1 到 PostTrainingV1.3 的唯一合并参考，面向后续 AI 或人工把其他分支合并到 main。它记录每一阶段的实际修改、生产运行时边界、模型和校准参数保护规则、GitHub checkout 后的 Docker 构建路径，以及冲突解决和提交前验证算法。

本文基于仓库代码、配置、Dockerfile、验证脚本、测试和既有提交记录编写，不依赖本机数据集、缓存、外部下载或被禁止阅读的比赛 docx。若其他文档与本文的生产约束冲突，以当前 Dockerfile、configs/RGBonly.yaml 和 docker/verify_submission.py 为准。

## 2. 版本链与变更边界

| 版本 | 提交 | 实际变更 | 合并结论 |
|---|---|---|---|
| PostTrainingV1 | d450e15 | 建立 Stage 3 HiT-Small 生产推理、RGB-only competition、球面控制器、校准产物和提交 Docker 结构 | 后续所有变更的基线，不得回退生产接口 |
| PostTrainingV1.1 | c14e70c | 修复 GitHub/Docker context、严格 checkpoint 加载、训练模块惰性导入和源代码验证 | 保留模型结构和参数，拒绝旧 checkpoint 别名 |
| PostTrainingV1.2 | 964a736 | 切换评测要求的 CUDA 12.8/PyTorch 2.11 栈并压缩最终镜像 | 最多 10 个 RootFS layer，当前实现 7 个 |
| BFoV 几何修复 | 227ca0f | 修复跨 -π/+π 经线的水平 BFoV 拟合，并跳过单个无效几何候选 | 不改变模型、校准或结果格式 |
| PostTrainingV1.3 | 本次新增提交 | 增加本指南 | 提交到远端 main |

从 d450e15 到 227ca0f 共涉及 23 个后续修改文件，约 642 行新增、132 行删除（不含本文）。V1 基线本身还包含完整的 HiT、训练、控制器、评测和文档体系；后续版本主要解决可复现构建、CUDA 兼容、镜像体积、模型加载保护和线上球面几何异常。

## 3. 不可破坏的生产合同

### 3.1 评测环境

- NVIDIA driver 580；
- RTX 5090 D v2，单卡，24 GB 显存，计算能力 sm_120；
- 16 CPU、64 GB 内存；
- CUDA 最高 12.8；
- 基础镜像必须是 pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel；
- Python 3.12、PyTorch 2.11.x、torchvision 0.26.x、CUDA runtime 12.8。

旧的 CUDA 12.4、PyTorch 2.6、torchvision 0.21 组合不属于当前生产合同，不得因冲突而恢复。

### 3.2 无参数 competition 入口

镜像入口固定为：

~~~dockerfile
ENTRYPOINT ["python", "/app/track.py"]
~~~

track.py 调用 instatarget.app.competition.runCompetition()。默认路径如下：

- 数据集：/mnt/dataset（DATASET_DIR 可覆盖）；
- 结果：/mnt/result（RESULT_DIR 可覆盖）；
- 配置：/app/configs/RGBonly.yaml（CONFIG_PATH 可覆盖）；
- HiT vendor：/app/src/instatarget/vendor/hit（HIT_ROOT 可覆盖）。

评测启动必须不传模型、序列或初始化框参数：

~~~bash
docker run --rm --gpus all --network none \
  -v <dataset>:/mnt/dataset:ro \
  -v <result>:/mnt/result \
  <image>
~~~

每个序列目录必须包含一个 .mp4 和 init.txt。程序为每帧写一行 clon,clat,fov_h,fov_v，先写 .partial，帧数校验成功后原子替换为最终结果。无效结果使用四个零值，不能改变官方文本格式。

### 3.3 模型与校准参数

固定生产文件：

~~~text
models/hit_small_stage3_inference.pth
models/hit_small_stage3_inference.calibration.json
configs/RGBonly.yaml
src/instatarget/vendor/hit/configs/HiT_Small.yaml
~~~

当前 checkpoint SHA-256：

~~~text
f9ee8e946f29a813ee359d5e417245651b622863759145abce82690e3fc12c66
~~~

严格加载参数总数：11,113,982。校准 JSON 的 checkpointSha256 必须匹配该值；YAML 中的 candidateMinScore=0.597262、fusionSourceMinConfidence=0.740642 必须分别与校准 JSON 相同；appearanceInput 必须是 presence_quality_product，SingleScore 权重为 0.50/0.50。当前校准文件 SHA-256 为：

~~~text
5d335beecfd6d46b1742655f742b7d5443001cdf3aea99425809fdf63807e68d
~~~

任何权重、模型结构、HiT_Small.yaml、校准 JSON、阈值、校准字段或参数名改动，都必须视为重新训练/重新校准。没有重新生成配对校准、重新计算哈希、重新严格加载和重新评测时，AI 必须拒绝该冲突版本。

## 4. PostTrainingV1 基线能力

V1 建立了后续版本必须继续使用的生产骨架：

- RGB-only 单目标跟踪和官方 competition 入口；
- 内置 HiT-Small vendor、Stage 3 推理权重和 calibration artifact；
- pytorch_hit_session.py PyTorch 后端；
- spherical geometry、BFoV 投影、运动预测、候选融合、状态机和结果 sink；
- configs/RGBonly.yaml 严格 schema 校验；
- Stage 3 的 presence、quality/predictedIoU 和 bbox 输出，外观原始分为 presence * quality；
- 紧凑 checkpoint、哈希绑定 calibration 和 Docker 验证脚本；
- 对应的训练、评测和运行文档。

V1 中新增的训练代码、HiT vendor 代码和模型配置仍属于仓库；生产 Docker 通过 .dockerignore 只保留运行时 import graph。

## 5. PostTrainingV1.1 的修改

### 5.1 Docker context 与 GitHub 可复现性

.dockerignore 不再排除整个 src/instatarget/training/，而采用白名单例外：

~~~text
!src/instatarget/training/__init__.py
!src/instatarget/training/model.py
~~~

生产复用 HiTTrainingModel，但不需要训练数据集、增强、manifest 或训练脚本。模型目录仍整体排除，只重新包含压缩 checkpoint、校准 JSON 和 models/README.md。

docker/verify_submission.py 新增：

- 必需 Docker 输入存在且未被 .dockerignore 排除；
- 必需输入已被 Git 跟踪，保证 GitHub checkout 可复现；
- checkpoint 小于 GitHub 100 MB 限制；
- checkpoint SHA-256 与 calibration JSON 一致；
- Dockerfile 复制源码、RGB-only 配置、两个模型产物和 track.py；
- 构建后在无网络容器中导入 competition 并严格加载 checkpoint；
- 运行时导入不能隐式加载 instatarget.data。

验证器支持文件、目录、通配符和 ! 反选规则。修改 ignore 文件必须重新运行验证器。

### 5.2 严格模型加载与惰性训练导入

pytorch_hit_session.py 把模型构建提取为 _constructRuntimeModel()，新增 validateHiTCheckpoint()：

- CPU 构造相同的 HiT-Small runtime；
- 读取 checkpoint 的 model state；
- load_state_dict(..., strict=True)，拒绝缺失或多余参数；
- 返回严格加载后的参数总数；
- Docker build 和 build 后 smoke test 都调用该函数。

src/instatarget/training/__init__.py 改为惰性导出训练数据 API，避免导入 runtime 时提前加载数据集、OpenCV 评测或开发依赖。因此生产 context 只需保留 __init__.py 和 model.py。

### 5.3 文档、配置和测试同步

同步更新 README.md、Competition/Controller/Tracker/User 文档、configs/RGBonly.yaml、模型说明及单元测试。生产文件名切换为 hit_small_stage3_inference.*；旧 hit_small_stage3.pth、旧 calibration 别名和旧网络状态不是兼容别名。

## 6. PostTrainingV1.2 的修改

### 6.1 CUDA 12.8 与依赖固定

~~~text
Python 3.12
torch==2.11.0
torchvision==0.26.0
CUDA 12.8
numpy==2.2.6
opencv-python-headless==4.11.0.86
PyYAML==6.0.2
timm==0.5.4
easydict==1.13
~~~

torch/torchvision 由基础镜像提供，docker/requirements.txt 不重复安装它们；其他依赖使用 --no-deps 安装。构建期断言 Python、PyTorch、torchvision、CUDA 和 sm_120。

### 6.2 镜像瘦身

基础镜像使用 devel 变体以满足评测栈，但最终 competition 镜像只保留预编译运行时。清理内容包括：

- /opt/pytorch、/opt/nvidia；
- /usr/local/cuda、/usr/local/cuda-12.8 编译工具链；
- Nsight、CMake、CUDA Python bindings、Triton；
- torch headers、share、protoc；
- NVIDIA wheel 的 include 目录；
- Python 缓存、文档、man、静态归档和临时文件；
- torchaudio。

必须保留：

- PyTorch/torchvision 预编译 runtime；
- NVIDIA wheel runtime 动态库；
- torch/bin/torch_shm_manager；
- NumPy 所需的 numpy._core.tests；
- OpenCV、PyYAML、NumPy、timm、EasyDict 和项目源码。

清理后使用 docker/partition_image.py 将有效文件系统复制到 scratch 最终阶段的 7 个互斥 bucket。当前镜像约 4.282 GB、7 个 RootFS layer。规范是 RootFS layer 数量 1 <= N <= 10，不是必须严格 7 层；7 层只是当前稳定布局。

### 6.3 验证器扩展

docker/verify_submission.py 还检查：

- 固定基础镜像和 CUDA 12.8 断言；
- RootFS layer 数量在 1 到 10 之间；
- 镜像大小不超过 5,000,000,000 bytes；
- 无网络运行时导入 smoke test；
- 严格 checkpoint 加载、sm_120 架构和不加载 instatarget.data。

最终 stage 当前仍要求按 00 至 06 顺序 COPY 七个 bucket；若未来改为 8 至 10 层，必须同步调整 partition 脚本、Dockerfile、验证器和本文档，且不得超过 10 层。

## 7. 当前 BFoV 几何修复（227ca0f）

### 7.1 故障与根因

线上错误：

~~~text
GeometryError: horizontal BFoV span is invalid: 6.27088505394077
~~~

原实现使用 max(horizontalAngles) - min(horizontalAngles)。当框跨越 ERP 经线接缝、角度同时接近 -π 和 +π 时，实际很窄的圆周区间被误算为接近 2π，触发 BFoV 合法范围检查。

### 7.2 修复规则

src/instatarget/geometry/spherical_geometry.py：

- 水平角使用最小圆周区间：排序归一化角度，寻找最大空隙，取其补集作为最小包络；
- 垂直角继续使用普通线性区间；
- 用圆周中点和线性中点重新拟合 BFoV 中心；
- 拟合迭代从 4 次提高到 12 次；
- 最终继续强制 0 < horizontalFovRad, verticalFovRad < π。

src/instatarget/app/driver.py：

- 新增 _projectValidObservations；
- 每个 view/observation 单独捕获 GeometryError；
- 记录 sequence、frame、view 和原因到 stderr；
- 丢弃该候选但继续当前帧和后续序列；
- 不把一个坏投影升级为整个 118 序列退出。

该容错只针对投影几何异常。模型加载、视频解码、结果格式、配置和 CUDA 错误仍必须失败，不得静默吞掉。

### 7.3 回归测试

测试覆盖跨经线 ERP 框、线上错误形态、单个无效球面投影候选跳过及有效候选继续提交。当前完整测试为 131 passed，定向 Ruff 为 All checks passed!。

## 8. 生产文件与开发文件边界

### 8.1 必须进入 Docker context 的文件

验证器要求以下路径存在、被 Git 跟踪且不被 .dockerignore 排除：

~~~text
track.py
configs/RGBonly.yaml
models/hit_small_stage3_inference.pth
models/hit_small_stage3_inference.calibration.json
src/instatarget/app/competition.py
src/instatarget/tracker/pytorch_hit_session.py
src/instatarget/training/__init__.py
src/instatarget/training/model.py
src/instatarget/vendor/hit/configs/HiT_Small.yaml
~~~

实际 COPY src ./src 会带入被 Docker ignore 保留的完整生产 import graph；不要只根据最小列表手工删除其他 src 文件。

### 8.2 必须保持忽略的内容

开发数据、视频、输出、测试、文档、缓存、原始训练权重、压缩包、AirSim360 诊断入口和训练数据集不得进入最终构建上下文。.gitignore 负责忽略本机模型/引擎产物；src/instatarget/vendor/hit 是源码，不能按普通 model artifact 忽略。

修改 ignore 文件时必须检查：

1. GitHub checkout 后必需文件仍被 git ls-files 找到；
2. Docker context 仍能导入 competition、HiT runtime 和 training.model。

## 9. AI 合并其他分支到 main 的算法

以下流程强制按顺序执行，任何一步失败都不得提交。

### Step A：识别基线和范围

1. git fetch origin main；
2. 确认目标是 main，记录 git merge-base HEAD origin/main；
3. 用 git diff --name-status d450e15..HEAD 和目标分支 diff，分类为模型/校准、Docker、生产源码、测试、文档、开发工具；
4. 不读取或引入本机数据、缓存和禁止的 docx 作为依据。

### Step B：冲突优先级

发生冲突时按以下优先级解决，而不是按分支新旧盲选：

1. **模型与校准完整性。** 保留当前 main 的 checkpoint、配对 calibration、HiT_Small.yaml；除非任务明确是重新训练/重新校准，否则拒绝另一侧。
2. **评测合同。** 保留 CUDA 12.8、PyTorch 2.11、torchvision 0.26、Python 3.12、sm_120 断言、无参数 track.py、/mnt/dataset 和 /mnt/result。
3. **几何安全。** 保留最小圆周区间、垂直线性区间、12 次拟合和 _projectValidObservations 单候选容错。
4. **运行时 import graph。** 保留 .dockerignore training 白名单、惰性 training.__init__ 和严格 validateHiTCheckpoint。
5. **算法实验。** 只有在不破坏以上合同并有测试/评测证据时才合并，否则留在独立实验分支。

### Step C：冲突后的必做检查

~~~powershell
git diff --check
python docker/verify_submission.py
python -m pytest -q
ruff check src tests docker
~~~

如修改 Docker、依赖、ignore、模型加载或配置，必须重新构建：

~~~powershell
docker build --pull=false -t instatarget:submission .
python docker/verify_submission.py --image instatarget:submission
~~~

必须用 --network none 做镜像 smoke test；不得挂载本地源码或依赖宿主机 checkpoint。GPU competition smoke 至少确认无参数启动、能读取数据、能写入 /mnt/result，并且跨经线候选不会退出。

### Step D：结果审计

提交前记录：

- git status --short 只包含预期文件；
- checkpoint SHA-256 和参数总数未变化；
- calibration JSON 与 YAML 工作点一致；
- Docker RootFS layer 数量 <= 10，当前应为 7；
- 镜像大小 <= 5,000,000,000 bytes；
- 全部测试、Ruff 和无网络 smoke 通过；
- competition 输出文件数量和每个序列帧数正确。

## 10. 明确禁止的操作

- 不得恢复 CUDA 12.4/PyTorch 2.6 旧栈；
- 不得把 devel 镜像的 CUDA compiler、Nsight、headers、CMake、Triton 带入最终镜像；
- 不得删除 torch/bin/torch_shm_manager 或破坏 numpy.testing 的 numpy._core.tests；
- 不得把层数检查改成“必须严格 7 层”；
- 不得把原始训练 checkpoint、优化器状态、数据集、视频或本机绝对路径写入镜像；
- 不得修改 checkpoint/calibration 后复用旧 hash、阈值和评测结论；
- 不得把单个 GeometryError 容错扩大到模型、解码、配置或输出异常；
- 不得关闭 strict checkpoint loading、hash match、无网络 smoke 或 Git-tracked 检查；
- 不得在 competition 启动时联网下载模型、依赖或配置；
- 不得仅凭 CPU 测试宣称 RTX 5090 D v2 可用，必须检查 CUDA 12.8 和 sm_120。

## 11. GitHub checkout 到评测的标准流程

~~~powershell
git clone <repository-url>
cd InstaTargetingSystem
python docker/verify_submission.py
docker build --pull=false -t instatarget:submission .
python docker/verify_submission.py --image instatarget:submission
docker run --rm --gpus all --network none -v "<dataset>:/mnt/dataset:ro" -v "<result>:/mnt/result" instatarget:submission
~~~

服务器必须能在无源码挂载、无本地模型路径、无网络的容器中完成构建后的运行。构建期网络只用于拉取已固定基础镜像和明确依赖；competition 运行期禁止网络。

## 12. PostTrainingV1.3 提交规则

本次提交只新增 docs/EnhancementInV1.md，不应包含模型、校准、Docker 或生产算法的意外改动。提交前执行：

~~~powershell
git diff --check
git status --short
git diff --stat d450e15..HEAD
git add docs/EnhancementInV1.md
git commit -m "PostTrainingV1.3"
git push origin main
git ls-remote origin refs/heads/main
~~~

远端 main 哈希必须与本地提交哈希一致。若推送被远端新提交拒绝，先停止，不得使用 --force；重新 fetch、审计差异并按本文优先级解决。

## 13. 快速提交前清单

- [ ] 未读取或依赖禁止的 docx；
- [ ] 目标分支是 main，提交名为 PostTrainingV1.3；
- [ ] Dockerfile 基础镜像为 pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel；
- [ ] PyTorch/torchvision/Python/CUDA/sm_120 断言保留；
- [ ] 最终 RootFS layer 在 1–10，当前布局为 7；
- [ ] 镜像大小不超过 5 GB；
- [ ] checkpoint hash、calibration、阈值和 11,113,982 参数未变化；
- [ ] .dockerignore 没有排除必需 runtime 文件；
- [ ] strict checkpoint load、无网络 smoke、完整 pytest 和 Ruff 全部通过；
- [ ] 跨经线 BFoV 回归和单候选 GeometryError 容错测试通过；
- [ ] 无参数 competition 能读取 /mnt/dataset 并写入 /mnt/result。
