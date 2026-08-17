# 帧事务协议

## 为什么需要事务

同一帧最多进行三轮查询。如果每轮都立即写入运动历史、模板或状态，后续轮次会在半更新状态上运行，并且异常后无法回滚。`FrameTransaction` 因此把一帧的所有尝试暂存到提交点。

## 协议顺序

1. `beginFrame(frame)` 创建 transactionId，并返回 attemptIndex=0 的 SearchPlan。
2. Runtime 必须按计划中的视图顺序返回 ProjectedObservation。
3. `consume(plan, observations)` 校验 sequenceId、frameIndex、transactionId、attemptIndex、stateRevision 和模板 expectedRevision。
4. 若证据不足，保存本轮 ProjectedObservation 并返回新的 SearchPlan；旧 plan 不能再次消费。
5. 后续轮把此前各轮和本轮的 ProjectedObservation 合成累计候选池，重新执行单框排序与两框融合。
6. 最终轮从累计候选池选择结果，调用状态机并一次性提交 Controller 内存，返回 FrameCommitted。

## 防止的错误

- 上一帧的后端响应迟到并污染当前帧。
- 同一 plan 被消费两次。
- 模板命令 revision 跳号。
- Round 1 结果尚未完成就并发开始 Round 2。
- 输出预测框被当作真实测量加入运动历史。

## 实现位置

协议类型位于 `core/protocols.py` 和 `core/types.py`；事务数据位于 `controller/state_model.py::FrameTransaction`；严格校验和提交位于 `controller/depth_aware_track_controller.py`。

未来并行化 Runtime 时，必须以 transactionId 和 attemptIndex 作为响应关联键，不能只依赖 viewId。

