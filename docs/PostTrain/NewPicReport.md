# 实验二：受控更换 HiT 目标模板

## 试验范围

本轮使用 `E:\tringData\shared_control\production2` 的 5 组 validation sequence：
`train_real/seq_0005`、`train_real/seq_0018`、`train_real/seq_0036`、`train_sim/seq_0010`、`train_sim/seq_0045`。
基线为对应 production2 `report.json`；两变体使用同一配置、同一 checkpoint 和同一 calibration artifact。

实现约束已实际执行：frame-0 anchor 永久保留；候选只从已提交、`valid=true`、measurement accepted 的结果生成；下一帧 shadow 验证时仍公开 anchor 分支；通过后从再下一帧使用候选；候选与 anchor 的 circular ERP IoU 低于阈值即拒绝；连续 3 帧 UNCERTAIN 回滚 anchor；shadow 分支使用 Controller 深拷贝且不写正式运动历史。

## 精度对比

| Sequence | Baseline ERP IoU | Strict ERP IoU | Strict Δ | Relaxed ERP IoU | Relaxed Δ | Baseline loss | Strict loss | Relaxed loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train_real/seq_0005 | 0.3396 | 0.3398 | +0.0003 | 0.3518 | +0.0122 | 0.72% | 0.36% | 0.72% |
| train_real/seq_0018 | 0.2380 | 0.2350 | -0.0029 | 0.2225 | -0.0155 | 41.22% | 41.56% | 44.86% |
| train_real/seq_0036 | 0.3175 | 0.2752 | -0.0423 | 0.3129 | -0.0046 | 34.19% | 43.05% | 34.81% |
| train_sim/seq_0010 | 0.4704 | 0.4743 | +0.0039 | 0.4629 | -0.0075 | 0.00% | 0.00% | 0.00% |
| train_sim/seq_0045 | 0.1927 | 0.2539 | +0.0613 | 0.2539 | +0.0612 | 34.67% | 16.76% | 19.96% |

宏平均 ERP IoU：baseline `0.3116`，strict `0.3157`，relaxed `0.3208`。
按 8,134 个可见帧加权：baseline `0.2740`，strict `0.2712`，relaxed `0.2750`。

## 专用记录

- strict：候选生成 864，验证后提升 503，回滚 196，非 anchor 模板帧 1526，shadow 额外耗时 68.09 s。
- relaxed：候选生成 1356，验证后提升 980，回滚 291，非 anchor 模板帧 2186，shadow 额外耗时 106.87 s。
- 所有验证延迟固定为 1 帧；事件文件记录 PhotoChangeRate、候选状态、验证 IoU、anchor/候选差异、shadow forward 次数和分阶段耗时。
- strict 的 PhotoChangeRate 均值约为 0.20--0.31，relaxed 约为 0.19--0.33，说明差异指标来自实际 GPU 模板张量而非占位 RGB。

## 结论与是否可取

本方案的工程机制可取：时序、anchor 保护、双分支隔离、回滚和成本记录均已在真实 checkpoint、CUDA 推理和完整序列上生效，且后端单元测试 9/9、策略测试 2/2 通过。

但不建议当前直接全量启用动态模板。relaxed 宏平均略高于 baseline，然而加权平均只从 `0.27397` 提升到 `0.27495`，同时在 `real/0018` 和 `sim/0010` 下降，并使 `real/0018` loss 从 41.22% 升到 44.86%、`sim/0045` loss 从 34.67% 降到 19.96%（strict 为 16.76%）；shadow 成本约 106.87 s/5 组，relaxed 还产生 291 次回滚。strict 加权平均反而降至 `0.27120`，说明当前 state-score/top-2 条件不能可靠预测模板迁移收益。

建议保留本实现作为受控实验能力，默认仍使用 frame-0 anchor。若继续优化，应先针对 sequence 条件增加候选的外观变化、尺度变化和验证后的独立收益门槛，并要求在加权 ERP IoU、tracking loss 和 P95 延迟同时不劣于 baseline 后再考虑上线；本轮数据不足以支持默认启用 strict 或 relaxed。

产物目录：`artifacts/controlled_template/template_strict`、`artifacts/controlled_template/template_relaxed`；每组包含 report、candidates、timings 和 template_events。
