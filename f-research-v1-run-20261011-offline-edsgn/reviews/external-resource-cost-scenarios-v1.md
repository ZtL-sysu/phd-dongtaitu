# 外部基线资源前瞻预算（metadata-only）

本文件只对已记录的 16 个单批 train forward/backward resource probe 做情景外推，不是训练计划验收或真实 epoch 预测。probe 没有 optimizer step。读取范围限于 probe JSON、aggregate_budget 输出和 NPZ 中的 `metadata_json`；没有打开数值数组、预测、测试 target 或 checkpoint。

每格展示实际单批 probe 的 forward/backward 秒数、batch size，以及由数据 split metadata 公式推得的完整 train/validation origin window 数。校验 probe status=passed、optimizer_steps=0、finite prediction/gradient 为真。

| 数据集 | 方法 | B | train origins | val origins | probe FB (s) | probe whole wall (s) |
|---|---|---:|---:|---:|---:|---:|
| email-episode | AGCRN | 4 | 334 | 46 | 2.293 | 2.422 |
| email-episode | DCRNN | 2 | 334 | 46 | 1.088 | 5.091 |
| email-episode | GraphWaveNet | 2 | 334 | 46 | 0.520 | 0.682 |
| email-episode | STAEformer | 2 | 334 | 46 | 0.554 | 0.737 |
| email | AGCRN | 4 | 528 | 74 | 2.474 | 2.663 |
| email | DCRNN | 2 | 528 | 74 | 1.144 | 6.397 |
| email | GraphWaveNet | 2 | 528 | 74 | 0.535 | 0.753 |
| email | STAEformer | 2 | 528 | 74 | 0.586 | 0.793 |
| mooc | AGCRN | 32 | 452 | 49 | 0.547 | 0.665 |
| mooc | DCRNN | 32 | 452 | 49 | 1.707 | 3.081 |
| mooc | GraphWaveNet | 32 | 452 | 49 | 0.450 | 0.605 |
| mooc | STAEformer | 32 | 452 | 49 | 0.321 | 0.472 |
| synthetic | AGCRN | 64 | 16112 | 4028 | 0.711 | 0.881 |
| synthetic | DCRNN | 64 | 16112 | 4028 | 1.624 | 2.941 |
| synthetic | GraphWaveNet | 64 | 16112 | 4028 | 0.675 | 0.898 |
| synthetic | STAEformer | 64 | 16112 | 4028 | 0.458 | 0.641 |

读取时的累计预算账本：spent 3.27 h / 72 h，remaining 68.73 h；coverage_complete=True，gaps=0，accounting_sources=88。这是账本快照。

情景成本按完整 16 个 dataset×method cells、相同 batch size 与 5 个模型 seed 线性叠加；10/50/100 epoch 数是情景输入，不是预测。

| epoch 情景 | 1 seed：train FB-only (h) | 1 seed：train+val whole-probe proxy (h) | 5 seeds：train FB-only (h) | 5 seeds：train+val proxy (h) |
|---:|---:|---:|---:|---:|
| 10 | 6.66 | 16.76 | 33.31 | 83.79 |
| 50 | 33.31 | 83.79 | 166.53 | 418.96 |
| 100 | 66.61 | 167.59 | 333.06 | 837.93 |

这里的 train FB-only 估算用 `ceil(train_windows/B) × 单批 forward_backward_seconds × epochs`；train+val whole-probe proxy 把每个 train 与 validation batch 都按整个 probe 的 wall_seconds 计价，作为较保守情景。两者都假设单批吞吐可扩展至完整 epoch，尚未实测；whole-probe proxy 也不是可证明的上界。

单批 probe 不能代表完整训练成本，也不能推断实际训练 epochs。早停、epoch 验证实现、模型状态保存、日志、GPU争用、seed间差异、错误重试与启动/结束成本均未知。结果只做预算风险 preflight，不据此移除任何必做 baseline。
