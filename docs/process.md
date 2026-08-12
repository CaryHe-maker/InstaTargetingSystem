# InstaTargetingSystem 运行流程与线程分配

> 本文档定义逐帧处理顺序、线程职责、队列协议和停止语义。  
> 单目标跟踪存在帧间依赖，系统只并行无状态工作；球面状态更新严格按帧序执行。

---

## 1. 总体流程

```text
AirSim360 / Raw panoramic video
          |
          v
   +-----------------+
   | T1 DecodeWorker |
   +--------+--------+
            | FramePacket
            v
   [decodeQueue, cap=3]
            |
            v
+------------------+   SearchRequest   +------------------+
| T0 ControlThread |------------------>| T2 InferWorker   |
| DTC / gate       |<------------------| RGB/RGB-D HiT    |
+--------+---------+   InferResponse   +------------------+
         |
         | ResultPacket
         v
   [resultQueue, cap=32]
         |
         v
   +-----------------+
   | T3 SinkWorker   |
   +-----------------+
```

默认使用四个长生命周期线程。一个帧事务最多向 `T2` 发送两次有界 batch，不为候选创建线程。
`T2` 负责局部 RGB 或 RGB-D 匹配、深度预处理、深度分支、融合头和模板执行；`T0` 负责窗口状态、
多视图计划、单帧候选聚合、模板命令和最终门控。Depth-to-RGB 保持在 `T2` 的 TrackerBackend
内部，geometry 只提供对齐的 RGB/Depth 局部视图。

---

## 2. 线程职责

| 线程 | 所有权 | 主要职责 | 禁止事项 |
|------|--------|----------|----------|
| `T0 ControlThread` | `TrackState`、状态机、球面运动状态 | 多帧预测、搜索规划、回投影、模板更新决策、最终门控 | 直接操作 CUDA/HiT 会话 |
| `T1 DecodeWorker` | 解码器 | 顺序解码、颜色转换、生成 `FramePacket` | 丢帧、修改跟踪状态 |
| `T2 InferWorker` | HiT 会话、设备流、模板特征 | 局部 RGB/RGB-D 推理、深度融合、批处理模板命令、返回 `LocalObservation` | 自行改变状态机、全局恢复或输出最终结果 |
| `T3 SinkWorker` | 输出文件、指标累加器 | 顺序写结果、可选可视化、统计耗时 | 阻塞控制线程、改写预测 |

设备后端只允许 `T2` 访问。PyTorch、ONNX Runtime 或 TensorRT 会话不得跨线程调用。

---

## 3. 逐帧时序

### 3.1 初始化帧

```text
T1                  T0                         T2                  T3
 | Frame(0)          |                          |                   |
 |------------------>| validate init bbox       |                   |
 |                   | bbox -> BFoV             |                   |
 |                   | aligned depth optional   |                   |
 |                   |---- InitRequest -------->| crop template     |
 |                   |<--- InitResponse --------| cache template    |
 |                   |---- Result(0) -------------------------------->|
 |                   | TrackState = TRACKING    |                   |
```

首帧结果必须等于校验并规范化后的初始框，不运行预测。

### 3.2 正常跟踪帧

```text
T1                  T0                         T2                  T3
 | Frame(n)          |                          |                   |
 |------------------>| update window state      |                   |
 |                   | predict center/FOV       |                   |
 |                   | build state-aware views  |                   |
 |                   |---- SearchRequest ------>| crop + batch HiT  |
 |                   |<--- InferResponse -------| RGB/RGB-D scores  |
 |                   | StateEvaluator           |                   |
 |                   | optional one escalation  |                   |
 |                   | update TrackState        |                   |
 |                   |---- Result(n) ------------------------------->|
```

`T0` 只有完成第 `n` 帧状态提交后，才为第 `n+1` 帧生成搜索计划。  
预测必须使用最近 `n` 帧窗口，不得用单帧直接决策。

### 3.3 低置信与恢复

1. `TRACKING/UNCERTAIN` 使用主视图与四个角保护视图；恢复使用环搜，丢失扫描使用六面 cube-map。
2. `T2` 将一次尝试的视图堆叠为 batch；显存不足时可确定性分批，但不能改变结果顺序。
3. `T0` 回投影后由 `StateEvaluator` 计算完整候选证据、球面连通簇和稳健融合框。
4. 不相交候选不会做 ERP 并集；支持不足的候选只作为下一次搜索种子。
5. 第一次为弱证据/拒绝且预算允许时，T0 可针对同一帧再请求一次；第二次必须提交。
6. `maxViewsPerFrameTotal` 约束两次尝试的视图总数，每个输入帧仍只发布一个结果。
7. 找回帧清空旧未来假设并重建运动窗口；深度无效时自动退化为 RGB-only。

恢复搜索预算由 `maxViewsPerFrameTotal`、`maxAttemptsPerFrame` 和 `globalSearchInterval` 共同限制。

---

## 4. 队列分配

| 队列 | 方向 | 容量 | 满时行为 | 元素 |
|------|------|------|----------|------|
| `decodeQueue` | `T1 -> T0` | 3 | 阻塞生产者 | `FramePacket` |
| `inferRequestQueue` | `T0 -> T2` | 1 | 阻塞生产者 | `InitRequest` / `SearchRequest` |
| `inferResponseQueue` | `T2 -> T0` | 1 | 阻塞生产者 | `InitResponse` / `InferResponse` |
| `resultQueue` | `T0 -> T3` | 32 | 阻塞生产者 | `ResultPacket` |

比赛模式禁止丢帧。小容量队列提供背压并限制高分辨率帧的内存占用。队列传递只读帧句柄；
最后一个消费者释放缓冲区。

---

## 5. 组件与线程归属

| 组件 | 运行线程 | 状态类型 |
|------|----------|----------|
| `AirSim360DataSource` | `T1` | 有状态、线程独占 |
| `SphericalMotionModel` | `T0` | 有状态、线程独占 |
| `DTC` | `T0` | 有状态、线程独占 |
| `RecoveryPlanner` | `T0` | 纯计算 |
| `DepthProcessor` | `T2` | 局部深度摘要和有效性 |
| `FusionHead` | `T2` | RGB/Depth 后端融合和 `fusedScore` |
| `BfovProjector` CPU 后端 | `T0` 或 `T2` | 无状态 |
| `BfovProjector` GPU 后端 | `T2` | 设备独占 |
| `HiTBackend` | `T2` | 有状态、设备独占 |
| `TrackStateMachine` | `T0` | 有状态、线程独占 |
| `ResultWriter` | `T3` | 有状态、线程独占 |
| `OtbEvaluator` | `T3` | 有状态、离线模式启用 |

CPU/GPU 投影由配置选择，但同一次运行只能启用一个实现。

---

## 6. 状态提交规则

`TrackState` 使用单写者模型，仅 `T0` 可修改。每帧按以下顺序提交：

1. 验证响应的 `sequenceId`、`frameIndex`、`stateRevision`、`transactionId` 和 `attemptIndex`。
2. 回投影全部局部观测。
3. 回投影全部局部观测，基于后端 `fusedScore`、运动连续性、尺度变化和可用深度一致性聚合候选。
4. 生成 `StateObservation` 并执行纯状态转移；必要时只推进帧内尝试，不提交跨帧状态。
5. 更新球面运动、目标尺度和丢失计数。
6. 生成模板命令；命令由下一次请求携带给 `T2`。
7. 最终尝试以 copy-on-write 替换控制内存，`stateRevision += 1`，发布只读 `ResultPacket`。

旧 revision/transaction/attempt、重复帧或乱序响应一律作为内部错误，禁止静默采用。

---

## 7. 模板更新线程协议

模板数据由 `T2` 持有，更新决定由 `T0` 持有：

```text
T0: TemplateCommand(kind, frameIndex, localBox, expectedRevision)
                             |
                             v
T2: verify backend revision -> crop feature -> atomic slot replace -> acknowledge
```

- `KEEP`：不更新。
- `UPDATE_RECENT`：替换近期模板。
- `UPDATE_STABLE`：替换稳定模板。
- `RESET_TO_ANCHOR`：清除动态模板，保留首帧模板。

模板替换在一次推理请求的边界执行，推理中途不得修改模板槽。
Controller state revision 每个提交帧增加一次；Backend template revision 每次推理尝试增加一次，
同帧升级后两者不要求数值相等。

---

## 8. 内存与设备资源

1. 解码缓冲区使用固定池，尺寸变化时重建并清空队列。
2. BFoV 输出张量按最大 batch 预分配并复用。
3. `T2` 使用单一设备 stream；测量耗时时显式同步。
4. 模板特征常驻设备，ERP 原帧在请求完成后释放。
5. 可视化帧默认关闭；开启时复制低分辨率图给 `T3`。
6. OOM 时先减小恢复 batch，再顺序分批；不得跳过当前帧。

---

## 9. 启动与停止

### 9.1 启动顺序

1. `T0` 加载并校验配置。
2. 启动 `T3`，打开临时结果文件。
3. 启动 `T2`，加载模型并完成 warm-up。
4. 启动 `T1`，打开视频或 AirSim360 序列并解码首帧。
5. `T0` 执行初始化协议。

模型 warm-up 失败时不得启动解码主循环。

### 9.2 正常停止

1. `T1` 发送 `EndOfStream`。
2. `T0` 等待最后一帧提交，向 `T2` 发送 `Stop`。
3. `T0` 向 `T3` 发送 `Finalize(expectedFrameCount)`。
4. `T3` 校验帧数，原子发布最终结果文件。
5. 主线程依次 join `T1`、`T2`、`T3`。

### 9.3 异常停止

任一工作线程发送 `FatalError` 并设置共享停止标记。队列唤醒所有阻塞者；`T3` 保留带
`.partial` 后缀的临时文件，主程序输出诊断并返回非零。不得产生看似完整的结果文件。

---

## 10. 离线训练与评测进程

训练不复用实时线程模型：

- 数据加载使用 PyTorch `DataLoader` 多进程。
- 单 GPU 每进程一个训练器；多 GPU 使用 DDP，一卡一进程。
- 评测以视频为最小任务并行，不拆分同一视频的帧。
- 每个视频内部仍按帧串行，防止状态穿越。
- 汇总进程只合并最终指标，不修改单序列结果。

---

## 11. 并发测试门禁

- [ ] 输入和输出帧号严格连续且数量一致。
- [ ] 单线程与四线程模式逐帧结果一致。
- [ ] 队列容量为 1 时无死锁。
- [ ] 恢复 batch 分批前后候选排序一致。
- [ ] EOF、解码错误、推理错误均能完整退出。
- [ ] 模板命令 revision 错误可被检测。
- [ ] ThreadSanitizer 可覆盖的本地组件无数据竞争。
