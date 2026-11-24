# P1B 独立评审：离线 EDSGN 实验包

review_execution: independent-agent（单一评审代理依次应用五种专家视角；不是五位独立评审者。）

## 裁决

**Recommendation: revise，然后进入开发。** 冻结方案足以启动受控 P2 smoke、数据获取和源码审计；不宜把 P1A 的路径 pass 解读成正式实验已无条件解锁。没有发现必须替换 SNN 家族或放弃研究方向的致命风险。正式 P3A/P4 前应锁定预测基线决策，并修正“完整矩阵”与 72 小时预算的完成定义。

**目标与假设：** 本评审对象是博士离线工程/实验包，不评投稿 venue，不要求发表或正向结果。任务是固定节点活动量预测：email 收/发计数 L28、horizon 1/3/7；MOOC 活动计数 L24、horizon 1/6/24。核心家族保持稀疏置信动态图、快慢 LIF、关系调节旧邻居记忆和冻结权重下图/状态独立调度；ANN 是对照。工程完成、假设支持与实测加速分别判断。

**Normalized idea：** 在历史窗口严格因果、固定节点集合下，检验关系质量摘要是否能抑制噪声驱动的图变化，图变化是否能帮助保留或忘却旧邻居状态，以及冻结参数后的独立图/状态调度是否可在完整软件计时下形成准确率—成本边界。Synthetic 用已知切换边界诊断；email/MOOC 检查真实观测活动。允许每个假设被否证。

## 搜索依据与最近工作

按冻结输入审阅，未将私有方案文本用于外部查询。文献状态为**定向、部分覆盖**，沿用 [P1 文献图](p1-literature-map.md) 中的原文和代码核验，非穷尽 novelty search。venue 未指定且不属于本任务，因此 venue 分按研究受众适配而非投稿标准打分。

| 近邻工作 | 已覆盖及主来源 | 重叠风险 | 仍可检验的差异 |
|---|---|---|---|
| DGCRN, TKDD 2023；[论文](https://doi.org/10.1145/3532611)，[实现入口](https://github.com/LakeChuan/DGCRN) | 信号驱动动态图 + 静态图融合用于交通多步预测。 | “动态关系改善预测”已不构成独立新 claim；适配依赖老旧混合栈。 | 置信/缺失稳健稀疏关系、关系调节记忆和冻结后独立门控需有单独消融；适配后重新训练，不引用原论文数值。风险中。 |
| SpikeSTAG, 2025 arXiv；[论文](https://arxiv.org/abs/2508.02069) | 自适应矩阵、脉冲图聚合、LSTM/脉冲注意力门控、多变量多步预测。 | 最接近“预测+自适应图+SNN+多步读出”的整体组合；否定宽泛的首次组合 claim。风险高。 | 具体差异限于置信关系、关系调节邻居保留、冻结后独立选择性执行及其可证伪证据。官方代码可运行性待核。 |
| SpikySpace, 2026 arXiv；[论文和匿名代码入口](https://arxiv.org/abs/2601.02411) | 脉冲 state-space 多变量时间序列预测及内部选择扫描。 | 已覆盖脉冲序列预测与选择性状态计算；不能把一般 selective compute 当成新意。论文内部扫描不同于本方案冻结模型上的独立图/状态调度。 | 作为脉冲预测对照；其能耗/复杂度结果不能代替本项目的软件端到端计时。匿名代码需先验安装与许可。 |
| SpikeNet, AAAI 2023；[论文](https://ojs.aaai.org/index.php/AAAI/article/view/26034)、[代码](https://github.com/EdisonLeeeee/SpikeNet)；Delay-DSGN, ICML 2025；[论文](https://proceedings.mlr.press/v267/wang25dm.html)；ChronoSpike, LoG 2026；[arXiv](https://arxiv.org/abs/2602.01124) | 动态图脉冲表示/传播、可学习延迟记忆、可学习 LIF 与时序编码。原任务主要是节点分类。 | SNN 动态图、时序记忆和自适应膜动力学已有近邻；分类成绩不等于预测证据。风险中。 | 只有使用统一预测头和协议重训才可作为定量对照，否则用于定位机制家族。 |

**Novelty delta：** 可检验的差异不在“把动态图和 LIF 拼起来”，而在置信关系变化是否能调节旧邻居状态保留，以及冻结模型上的图/状态独立门控是否能以真实 indexed execution 形成准确率—成本边界。SpikeSTAG 和 SpikySpace 已压缩宽泛 novelty；当前可防守范围是机制交互、负结果边界和端到端执行证据。材料不支持原创性保证。

## 严重风险与专家视角

### Field expert

- **Score tendency 3/5；confidence 中。** 问题可测，H1–H3 可独立失败。拒绝级 concern：SpikeSTAG 已覆盖 adaptive graph + spiking temporal + multi-step forecasting；若贡献仍称为“首次 SNN 动态图预测”或仅称模块组合，会被 prior art 直接削弱。
- **Anchor / why：** P1 文献图的 SpikeSTAG、DGCRN。宽泛 claim 无法区分已知架构。
- **最小修复：** 贡献表逐项对照 SpikeSTAG 已有模块，只把置信关系、关系调节记忆、冻结后独立调度和检验方式列为待证差异；P3A 前更新近邻搜索。若差异只剩新数据集，明确收窄成适用边界研究。符合后 novelty 判断可上调约 0.5（不是接收概率）。

### Method expert

- **Score tendency 4/5；confidence 中高。** Mechanism card 和 engineering interface 对质量调节、旧邻居记忆、分离动作路径给了可实现定义；V3/V4、V1/V2、打乱 c、恒定 g、无条件重置等有明确反证用途。
- **风险与边界：** T01–T26 还未执行，不能从规格推断稀疏路径/梯度语义正确。relation row 的 ID 对齐、空/非空转变初始 c、延迟 reset 一次性消费与全开索引/参考等价必须由实际测试保证。候选控制风险在 engineering-interfaces 已修订：同 seed 共享探索日程，V9 固定 confidence-specific branch，只改置信度更新。
- **最小修复：** P2 逐项保存测试映射、稀疏 call counters、有限梯度、all-on equality、相同候选池/支持 hash。任何语义/泄漏测试失败都阻断假设结果解释；无需替换方法家族。

### Experiment expert

- **Score tendency 4/5；confidence 中。** 时间划分、目标 mask、训练期统计、五 seed、paired block bootstrap、Holm、checkpoint hash、周期/随机对照和实际 indexed replay 路径较完整。负结果有明确解释规则，不需要强行显著。尚未锁定的最近预测 SNN 对照是重要但可修复的缺口。
- **主要修复项 A：最近预测强基线要锁定。** execution/feasibility contract 明确 DGCRN；experiment-traceability 将 SpikeSTAG、SpikySpace 和 RP 预测模型列为 pending，但未给选择顺序、调参公平预算或不可运行时替代规则。若近邻脉冲预测模型全不跑，强结论只能是内部机制相对消融/DGCRN 的结果，不能说明相对现有 SNN forecaster 的表现。
- **主要修复项 B：72 小时与“complete experiment matrix”表述略有冲突。** traceability 清楚列出多个数据集、V0–V9、噪声/缺失/扰动、baseline 和 profile 条件；execution contract 要求执行 complete matrix，同时也承认超时可有 uncompleted rows。具体运行时尚未实测；当前 GPU 报告只有约 9441 MiB free 且 util 77%，其他作业不能停，不能按完整 12GB 空卡估算。
- **最小修复：** P2 不停其他任务，记录当时空闲显存、util、可用 CPU/RAM、单批完整训练/回放 profile；用测量估算预算内覆盖，不减少正式节点/测试/五 seed。P3A 前明确核心矩阵必跑与扩展矩阵可挂起，所有 traceability 行始终登记 run/未运行及原因；将“complete”定义为追踪矩阵全覆盖、核心实验必跑、扩展项未跑则显式 pending，不能称 pending 为完成。至少锁 DGCRN 和一个最近脉冲预测器（SpikeSTAG 优先，SpikySpace 备选）相同任务上的 run；若代码不可用，留下载/依赖/入口 smoke 证据，并在看 test 前锁定有理由的可重训 forecasting fallback。代码锁 commit、任务头差异和相同验证调参上限。

### AC / target-audience expert

- **Score tendency 3/5；confidence 低至中。** 按博士研究包看，工程正确性、假设结果和实际加速分别审计，适合形成可复核证据。未指定投稿 venue，故不能评 venue fit 或接收可能。
- **风险：** 最终若只有源码/运行日志而没有按 H1/H2/H3 给出适用及失败边界，会退化为集成工程；但论文和投稿并非当前交付要求，不构成 P1B blocker。
- **最小修复：** 每一假设报告支持/否证/未决，列 failure slice 和真实成本来源；保持不添加在线部署或额外任务，不需补作者/venue 信息。

### Skeptical prior-art expert

- **Score tendency 3/5；confidence 中。** SpikeSTAG、SpikySpace 和 DGCRN 限制了“动态图+SNN+forecasting”的宽 novelty；Delay-DSGN 和 ChronoSpike 限制“脉冲记忆自适应”的独立新颖表述。
- **最强反对：** 多个已知机制组装可能缺少超越模块消融的 insight；SpikeSTAG/SpikySpace 若因代码不便而豁免，将是选择性 baseline。
- **最小修复：** 复核最近 forecasting-SNN 文献的版本、查询词和主来源；P3A 锁至少一个已实际适配的 forecasting-SNN 强比较，或留下可重复失败证据及事先选定的 fallback。没有证据时将新颖性写成未知/有限，不以文献未检出为不存在。

## 评分表

按 1–5 rubric 评当前离线研究方案 readiness，不是论文接收概率。加权结果为 Σ(score×weight)/100 = **3.70/5.00**。

| 维度 | 权重 | 分数 | 信心 | 扣分/依据 | 修复条件 |
|---|---:|---:|---:|---|---|
| 问题重要性 | 12 | 4 | 4 | 非平稳活动、缺失/噪声和可避免刷新有明确定义，但结论域限于历史记录。 | 不外推组织/学习结果。 |
| 相对新颖性 | 14 | 3 | 3 | SpikeSTAG/SpikySpace/DGCRN 覆盖重要组合，检索为定向而非穷尽。 | 机制差异表、最近预测对照和检索更新；若只剩数据集应用则收窄 claim。 |
| 概念创新 | 12 | 4 | 3 | 机制交互与反证路径明确，但组件家族有近邻。 | 实验证明模块必要性及失败边界。 |
| 方法可靠性 | 14 | 4 | 4 | 固定宽度、因果输入、匹配控制、独立门控定义清晰；实现尚未通过 T01–T26。 | P2 测试和稀疏调用审计通过。 |
| 简洁性 | 8 | 4 | 4 | 三项假设可分别判断，未扩展至服务/在线学习。 | 不增加核心机制，报告完整成本。 |
| 资源可行性 | 8 | 3 | 3 | RTX 3060/本地 Conda 与72小时上限已知；当前 free9441MiB、util77%，全矩阵时长未测。 | P2 在不停止其他任务下实测并冻结预算分层。 |
| 实验说服力 | 10 | 4 | 3 | 完整 traceability 已列 B1–B4、H1–H3、鲁棒性、统计与 profile 路径，强预测/脉冲基线已纳入必查；尚 pending 且需正式锁选择。 | P3A 前锁 DGCRN+至少一个近邻 forecaster 或有证据的 fallback。 |
| Venue/受众匹配 | 8 | 3 | 2 | venue 未指定且当前不是投稿任务；可评博士实验路径，不能评 venue taste。 | 将来若投稿再单独评 venue；当前不需要补信息。 |
| 时效性 | 6 | 4 | 3 | 文献图覆盖 2023–2026 关键脉冲/动态预测工作，正式 freeze 前仍需更新。 | P3A 前更新近邻检索。 |
| 风险调整后潜力 | 8 | 4 | 3 | 假设可证伪、允许负结果、数据与本机路径已盘点；基线和资源证据尚未落实。 | 完成下方门槛后进入正式实验。 |

所有 ≤3 分均在本表给出具体依据与条件；低信心单独标示。

## 最小修复门槛

| ID / 问题 | 严重度 | P2/P3A 可检查条件 | 预期评分影响 |
|---|---|---|---|
| P1B-1 最近预测基线选择 | 高；可修复 | P3A 记录 DGCRN commit/license/改造差异，并至少实际适配运行一个最近脉冲 forecaster：优先 SpikeSTAG，若代码/依赖不可用可用 SpikySpace。要保留失败日志和可核验 fallback；统一数据切分、输入/H、目标变换、验证选择和调参上限。 | 实验说服力可由3升4。 |
| P1B-2 资源与矩阵完成口径 | 高；P2 gate | P2 在 GPU free9441MiB/util77% 类实际共享负载下测量每类训练和 replay 的时间/峰值显存；不停止其他作业。P3A 前将 traceability 标记核心必跑和扩展可挂起，显式逐行记录 pending。execution contract 中“完整矩阵”不得与72小时截断冲突；未跑项不能算完成。保留全节点/五 seed/测试覆盖。 | 可行性可由3升4；若 core 超预算则返回重排而不偷减覆盖。 |
| P1B-3 候选/机制对照实现验收 | 中；已有协议补足 | engineering-interfaces 已规定同 seed 共享探索日程、确定性 ID tie、固定候选池；P2 存候选池/支持 hash，确认 V9 只关闭 confidence-specific 分支。H2 controls 共享作用时刻与宽度；T 测试记录 ID 对齐、空行语义、一次性 reset、all-on 等价。 | 保持 soundness 4；失败则停止主张归因并修复实现。 |

## 评审综合与状态

- **共识：** 离线数据契约、因果隔离、匹配消融和不预设正结果的判定规则具体；负结果不构成 fatal risk。
- **分歧：** 方法视角认为三模块可实现且可识别；prior-art/实验视角认为最近预测 SNN 尚未变成已锁定的运行比较。该差异不能靠内部消融平均掉。
- **决定性 accept 轴：** 任务契约与三个可否证主张明确，工程和结果边界可审计。
- **决定性 reject 轴：** 正式实验启动前的最近预测 baseline 和有限资源下的 core/extension 完成口径未锁。
- **Confidence：中。** 冻结协议与论文原文可检查；本轮未重新下载 SNAP 数据或执行 GPU profile/T 测试，SpikeSTAG 可运行性、实际时长和穷尽 prior art 仍未知。新观测资源为 RTX 3060 12GB、free 9441MiB、util 77%、CPU12线程/RAM31.9GiB；不能停止其他任务。
- **最终建议：revise。** 两项高优先修复和 P2 smoke 完成后，建议 accept-to-develop；无需更改用户要求保留的 SNN/核心家族。不是 abandon，也不是新颖性保证。
- **Checklist：standard；** 已评 normalized idea、近邻、五种专家视角、十维评分、fatal/fixable、置信度、预算/测试/数据/候选公平与研究路径。venue 维度因任务范围不适用投稿审核；未运行代码或实验。
- **Output self-check：** 分数≤3均有证据与修复条件；表格完整；未声称多位独立专家；没有接收概率、预设性能百分比或把假设负结果视作致命。

