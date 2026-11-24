# P1：可检查文献与基线路径核验

**范围。** 离线固定节点活动量多步预测：email 日收/发量，L=28、H=1/3/7；MOOC 课程活动小时次数，L=24、H=1/6/24。目标方法包含稀疏置信动态图、关系调节快慢 LIF、冻结权重下独立图/状态调度。本文件只核验近邻方法与基线路径，不给出新颖性保证，也不以本次小范围未检出推断文献不存在。计划段落明确指出这两组预测定义属于本研究适配协议，不是数据集原生 leaderboard task；SpikeNet、Delay-DSGN、ChronoSpike 原任务需 forecasting head/输入协议才可作量化比较（`inputs/rp-extracted.txt` §5.2、§5.3；`inputs/implementation-plan.md` §2、§10）。

## 论文和可复现入口

| 工作 / 可检查来源 | 原任务与类别 | 与目标机制已知相近之处 | 可运行适配路径与公平性/依赖风险 |
|---|---|---|---|
| [DGCRN, TKDD 2023](https://doi.org/10.1145/3532611)；[论文出版社页](https://dl.acm.org/doi/10.1145/3532611)；[作者列出的原始 PyTorch 实现](https://github.com/LakeChuan/DGCRN) | 交通传感器多步预测；method + benchmark。每步由 hypernetwork 从节点属性生成动态 filter/embedding，形成动态邻接并与先验静态图结合，进 DGCRM/seq2seq。 | 最接近“从当前节点信号学习时变预测关系”；动态关系与时序更新耦合。 | 首选外部强预测基线。保留原图生成与DGCRM，在数据适配器上改输入为固定节点计数、输出对应多步非负计数，重训相同时间切分/调参预算。原仓库面向 METR-LA/PEMS-BAY及交通预处理，不等于邮件/MOOC复现；README列出 Python 3.6、旧 NumPy/Pandas、PyTorch、MXNet 1.4.1、TensorFlow 等组合，现代环境冲突风险高。优先冻结 commit、单独 conda 环境；不把删依赖/重写后的模型仍标作无差异官方复现。旧 `tsinghua-fib-lab/DGCRN` URL 当前失效；出版社页指向 `tsinghua-fib-lab/Traffic-Benchmark`，LakeChuan 仓库 README 自称 origin PyTorch implementation 且是该 benchmark fork。 |
| [SpikeNet, AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/26034)；[论文 arXiv](https://arxiv.org/abs/2208.10364)；[作者代码](https://github.com/EdisonLeeeee/SpikeNet) | DBLP/Tmall/Patent 等动态图节点分类；pure method。动态快照上的 SNN 图表示学习。 | 直接覆盖“脉冲神经元用于动态图表示/邻居传播”；不是 count forecaster。 | 适合作为分类上下文、或做适配工作包：沿用 encoder/传播，替换分类 readout 为 H×F count head，因果地构造每个时刻输入；用同一节点集合/切分重新训练。代码数据需 Dropbox/Aliyun 下载，README指出节点特征由 DeepWalk生成（部分数据可需分钟至小时），原始任务和数据管道与活动预测不同。只有加入预测头及统一预处理后的结果才可比较；论文节点分类结果不可与MAE/RMSE比。 |
| [Delay-DSGN, ICML 2025 / PMLR 267](https://proceedings.mlr.press/v267/wang25dm.html)；[论文PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/wang25dm/wang25dm.pdf) | 大型演化图节点分类；pure method。高斯延迟核按时间延迟邻居信息，降低历史遗忘。 | 与快慢 LIF 的历史保留/传播延迟最接近；都是动态脉冲图上的时间记忆机制。 | PMLR论文页核验了方法、任务与发表信息；本次未从论文主页核实到作者代码链接（检索未命中不代表不存在）。可按论文重实现时必须清楚标“论文实现/适配”，不能称官方复现；或作者代码可取得后加 forecasting head。延迟机制需要因果对齐预测原点，不可使用窗口之后的事件。原节点分类成绩不构成预测基线。 |
| [ChronoSpike, LoG 2026（arXiv版本4，2026-10-02）](https://arxiv.org/abs/2602.01124)；[作者关联代码仓库](https://github.com/Abrar2652/ChronoSpike) | 大型动态图节点分类；pure method（当前 arXiv 记录称 accepted at LoG 2026）。learnable LIF、逐通道膜动力学、连续特征多头空间聚合及轻量 Transformer 时序编码。 | 明确已有可学习/自适应膜动力学与时序图表征；局部空间聚合和多尺度/快慢脉冲状态有机制交叉。 | 可以将分类 readout 替换为 count forecasting head，再用指定 L/H 与因果滚动切分训练；代码仓库存在，但检查页显示无 README 描述、无 release，入口和环境需先审读/锁 commit。预测适配会改变论文原任务，须单列适配结果。版本较新，主张以指定 arXiv版本为准并复查变更。 |
| [SpikeSTAG, arXiv 2025](https://arxiv.org/abs/2508.02069)；[论文全文 HTML](https://arxiv.org/html/2508.02069) | METR-LA、PEMS-BAY、Solar、Electricity 多变量多步预测；pure method。自适应矩阵、spiking SAGE 多尺度聚合、LSTM 与 spiking self-attention 双路门控融合。 | 在目标清单中与“预测 + 自适应关系 + 脉冲时序处理 + 多步读出”整体最接近；且具有门控融合。 | 应列为最优先的外部脉冲预测候选。复用论文代码若能确认官方仓库；本次 arXiv全文和公开检索未核实作者代码URL，故先记录待查而不伪造入口。适配原生节点计数时必须重训输出头/归一化/时间特征；其全文用 minute/hour/day-of-week 等协变量，而两个目标数据的时间戳是相对时间，日历特征不可凭空生成。报告对照需去掉无效日历特征或替换为双方共用且因果合法的特征，并披露差异。论文报告的公开四套基准结果不等于本任务可比数值。 |
| [SpikySpace, arXiv 2026](https://arxiv.org/abs/2601.02411)；[作者提供代码（匿名归档）](https://anonymous.4open.science/r/SpikySpace) | 四个多变量时间序列预测基准；pure method。脉冲选择扫描的 state-space forecaster，主张线性序列复杂度。 | 同时覆盖 spiking temporal forecasting 与高效选择性状态更新；其“选择性”是状态空间内部运算机制，不是本方法的冻结权重独立图/状态调度器。 | 可直接把固定节点看成多变量序列的变量维，适配计数输入、H×N输出并重训；代码归档当前匿名，需检查依赖、模型训练入口和许可证/固定版本，不能预先保证可运行。原论文 benchmark/能耗设置与软件 wall-clock、离线日志回放不同；不要把论文理论/硬件能耗数当成本项目实测节省。 |

## 基线路径判断

1. **预测主基线优先级：DGCRN → SpikeSTAG → SpikySpace。** DGCRN 是成熟的动态图交通预测对照；SpikeSTAG 是同类“自适应图 + 脉冲时序 + 多步 forecasting”高相似风险；SpikySpace 则控制脉冲时间序列预测能力。三者都须在 email/MOOC 目标上重训，并共享节点、因果切分、输入长度、输出偏移、目标变换、调参资源与误差指标。按 email 的 H=7、MOOC 的 H=24 输出完整未来向量，评分抽取指定 horizon；适配后的方法名标记 `-adapted` 并记录 commit、改动和配置。
2. **机制背景/适配候选：SpikeNet、Delay-DSGN、ChronoSpike。** 这些工作已有动态图 SNN、脉冲传播或可学习记忆机制；它们的主任务为节点分类。最公平的路径是用同一 forecast head 和输入协议进行专门适配，并明确其目标函数、参数预算及代码差异；若不做适配，只作为机制相近文献，不能填入预测 leaderboard 表。
3. 保留实现方案已经规定的 persistence、合法周期均值、linear、compact recurrent 与内部连续状态匹配控制。相对时间日志不支持 weekday/holiday 等日历外生变量。对 MOOC 排除未来含义的 dropout label。以上约束来自 `inputs/implementation-plan.md` §2、§6、§10 和计划中的公平协议。

## 最重要实现风险

- **目标错配最危险：** 节点分类成绩不能冒充多步活动预测结果；不同任务的 head、损失、时间切分和指标都不同。SpikeNet/Delay-DSGN/ChronoSpike需新增 count forecaster，逐一标为适配版。
- **方法身份与代码来源：** DGCRN `LakeChuan/DGCRN` 可核实为作者/论文指向的原始 PyTorch 实现入口，但其 README 写明 fork 自 Traffic-Benchmark；应记录仓库与 commit，而非声称它是独立原始仓库。`tsinghua-fib-lab/DGCRN` 为失效旧路径。Delay-DSGN 和 SpikeSTAG 的作者代码本轮未能从论文主来源核实，状态记为“待确认”，不要把未检出写成不存在。
- **时间协变量泄漏/造特征：** 预测论文常依赖时钟/日历信息；相对 timestamp 无法恢复星期、节假日。统一关闭或用在数据集两者中都可由过去确定的特征。
- **效率主张不可照搬：** ANN/GPU上的稀疏门控可能因索引、scatter、动态图构造而比dense慢；SpikySpace/SpikeNet论文本身的能耗或复杂度论断不等于本软件的真实端到端运行时。必须同机、同数据、计入图构建与调度开销，另行报告训练和冻结模型推理。
- **依赖与可运行性：** DGCRN README 中旧 Python/NumPy/Pandas、MXNet、TensorFlow 堆栈有冲突风险；ChronoSpike仓库当前可见文件虽有模型脚本但未见可用安装说明；SpikySpace仓库为匿名代码归档；SpikeNet数据另行下载。先锁源码版本与环境，跑通 smoke test 再纳入定量对照。

**来源纪律：** 最终结论来自论文出版社/AAAI/arXiv主记录及项目自身仓库页面；搜索结果只用于定位，不用于单独支撑论文性质或方法结论。核验时使用的公开查询词包括 `DGCRN dynamic graph traffic forecasting official implementation`、`SpikeNet dynamic graph spiking AAAI`、`Delay-DSGN ICML`、`SpikeSTAG multivariate forecasting`、`ChronoSpike dynamic graph`、`SpikySpace time series forecasting`。本次仅为聚焦核验，未声称穷尽相关文献。
