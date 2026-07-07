这个方向很对——静态选择题最大的问题是可以被记忆/污染，而且"Architecture Intuition"本质上就是一个交互式技能：真正的研究员不是一眼看出答案，而是知道**该跑哪个实验、看哪个 observable、什么时候停**。ARC-AGI-3 的核心设计正是这个思路：不给说明书、给交互环境、拿人类当 baseline、按探索效率打分。我先把几个可以直接嫁接到 ArchitectureIQ 上的设计丢出来。

## 核心转变：从"答题"到"花预算买信息"

把每道题变成一个 episode：agent 有一个**探针预算**（比如总共 5000 training steps 或 10 次 query），可以自由分配去做部分训练、看曲线片段、查 observable，最后必须给出答案。评分不只看对错，而是画出 **accuracy vs. compute spent 的 Pareto 曲线**——一个真懂 learning mechanics 的模型应该用极少的探针就收敛到正确答案，一个不懂的模型只能靠暴力跑满。这直接把你们文档里"到底训练多少步"的 concern（2000 步和 16000 步 rank 相关只有 0.1）从缺陷变成了考点：**知道早期信号何时可信，本身就是被测能力**。

<br>

具体的 action space 可以很小但表达力很强：

- `train(candidate, hparams, steps, seed)` → 返回部分曲线
- `probe(run_id, observable)` → 花额外 cost 查 grad norm / weight norm / feature rank（不同 observable 定价不同，逼 agent 思考哪个诊断信号最有信息量）
- `intervene(run_id, action)` → 改 LR、加正则、resume（用于下面的"训练医生"任务）
- `answer(...)` → 结束 episode

## 几个具体的环境/游戏原型

**1. 黑盒架构鉴定（System Identification game）。** 环境里藏一个模型（CNN / Tiny Transformer / RNN / MLP 之一，架构和 config 不告诉你），agent 只能设计输入数据集去 probe 它、观察训练行为，用最少的 query 猜出架构族。这是最"ARC 味"的一个——本质是让 LLM 玩科学家版 20 questions，而且天然防污染，因为每局都是程序化生成的新黑盒。

**2. 训练医生（Training Doctor）。** 给一个"病态 run"——loss 平台、梯度爆炸、grokking 迟迟不来——agent 在 k 次干预内诊断并修复。这个任务对你们的 Task 7 传播面也最好：人类研究员玩这个会很上头，还能顺便收人类 baseline 数据用来 filter "human-easy, AI-hard" 的题（ARC 的选题铁律）。

**3. 预算分配锦标赛。** N 个 candidate、一个总 step 预算，agent 自己决定 early stopping 和加注，最后按找到最优模型的 regret 打分。这里有个很妙的点：**Hyperband / Successive Halving 就是现成的非智能 baseline**——如果 LLM agent 打不过一个不看任何语义信息的调度算法，说明它的 architecture prior 一文不值。这是很强的一条 headline claim。

**4. 在线滚动预测（World Model 模式）。** 训练一步步推进，agent 每隔 k 步预测下一段曲线，环境揭示真值，滚动打分 + calibration。这个和你们 NCPL 的 cold-start curve prediction 的洞察直接接上了：交互版就是"观测 0 → 1 → n 个 epoch 时预测质量如何演化"，把那个 gap 变成了 benchmark 的正式指标。

## 工程上让它跑得动的关键一招

ARC-AGI-3 的环境便宜是因为是格子游戏；你们的环境是真训练，每个 episode 都跑 GPU 不现实。解法是**预计算 lattice**：用 ComfyResearch 把 (dataset × model × hparam × seed) 的网格提前跑完存成查询表，交互时环境只做 lookup + 插值——这其实就是把 NAS-Bench 的思路反过来用：NAS-Bench 作为静态数据集太重没法描述（你们已经发现了），但**作为交互环境的后端它是完美的**，agent 每次 query 毫秒级返回。再进一步，稀疏格点之间可以用你们的 NCPL meta-model 做 surrogate 填充，留一个 held-out 真跑子集做校验——这样"训练 meta model"这条交付线和"环境套件"这条线就合并成一个东西了。

Seed variance 也顺手解决：lookup 表里每个格点存 n 个 seed，环境每次随机吐一个 draw，agent 想要显著性就得自己花预算重复采样——统计素养直接变成隐性考点。

## 一个值得先定的分歧点

在动手之前有个岔路要选：环境是**给说明书**（agent 知道 action space 和目标，纯考 architecture 直觉）还是 **ARC-AGI-3 式不给说明书**（连"这个环境在测什么"都要自己探索出来）？后者更酷、更防作弊，但会把 architecture 直觉和通用探索能力混在一起，归因变难。我的倾向是 V1.5 给说明书、把变量控干净，V2 再上 mystery 模式——这和你们文档里"故意选最简单完全可控"的哲学是一致的。

你们 7 月 3 日 DDL 的那两个环境（PDE、桌球碰撞）其实已经是 Task 6 的雏形了——最小的改造路径可能就是给它们加上 query budget 和 gym 式的 step/observe 接口，先跑通一个"预算分配锦标赛"，看 Opus/GPT 能不能打过 Hyperband。要不要我帮你把这个交互协议写成一份 `env_spec.md` 草稿，对齐你们 TODO 里说的上下游 protocol？

我最推荐的一个：**架构指纹（Architecture Fingerprinting）**——它是你们 Task 2（连线题）的反向版本，但难度和含金量完全不在一个量级。

## 旗舰候选：架构指纹

**规则**：给 agent 一组架构（比如 MLP / Tiny Transformer / RNN / CNN，config 公开），让它**设计一个数据集**，使得这 N 个架构在上面训练时产生的 learning curve 彼此**最大程度可区分**——理想情况下，只看曲线就能唯一反推出是哪个架构在训练。

**评分完全自动化**：环境拿 agent 提交的数据集真跑（或查 lattice），把 N 条曲线喂给一个固定的判别器（甚至简单到最近邻匹配），判别准确率 + margin 就是分数。不需要任何人工标注。

这个任务有意思的地方在于它逼模型做的事情，恰好就是"architecture prior"的操作性定义：

- 连线题（Task 2）只要求**识别**签名——"attention 的曲线长这样"，这是可以背下来的知识。指纹任务要求**制造**签名——你必须知道每个架构的 inductive bias 的**因果机制**，才能构造出一个刺激让机制显形。比如想把 Transformer 和 RNN 分开，你得知道往数据里塞长程依赖 + 可变长度；想把 MLP 和 CNN 分开，得知道打破/保留平移不变性。答对的唯一路径是懂机制。
- 它天然是**组合爆炸防污染**的：架构池每局随机抽，config 随机扰动，网上不存在任何可背的答案。
- 有一个非常干净的失败信号：不懂的模型会提交"generic 难数据集"——所有架构都学得慢但曲线形状雷同，判别器分不开，得零分。**难 ≠ 可区分**，这个 gap 正是 prior 和瞎猜的分界线。

进阶版还可以加一个漂亮的约束：**数据集大小/复杂度预算**。用 1000 个样本就把四个架构分开的 agent，比需要 10 万样本的 agent 拥有强得多的 prior——这又回到"用最少信息买最大区分度"的主旋律，和你们交互式预算的框架无缝衔接。

## 备选一：架构叠叠乐（Architecture Jenga）

给一个**过度工程化**的模型（该有的不该有的组件全堆上：residual、LayerNorm、attention、positional encoding、gating……）和一个数据集，agent 每回合抽掉一个组件，只要 retrain 后性能不掉出阈值就继续。**得分 = 成功抽掉的组件数**。

关键设计是数据集决定哪些组件是承重墙：modular addition 上 attention 拆了就塌，XOR 上非线性是命根子，long-range copy 上 positional encoding 动不得——但在别的数据集上这些可能全是冗余。所以高分要求的是 **(dataset × component) 的联合理解**，而不是"residual 一般有用"这种泛泛的背诵。

它的附加价值是**人类可玩性最强**——这就是给 ML 研究员玩的叠叠乐，直接喂你们 Task 7 的传播需求，还顺手收人类 baseline。而且贪心策略（每次拆完真跑一遍验证）是天然的非智能 baseline：真有 prior 的 agent 应该敢**一次性拆一串**，用更少的 retrain 预算达到同样的拆除数。

## 备选二：反事实手术（Counterfactual Surgery）

给 agent 一次**完整跑完的训练记录**（run A 的全部曲线和 observable），然后问：如果只改一处——把 ReLU 换成 sigmoid / 删掉 residual / LR 乘 10 / 去掉 weight decay——曲线会怎么变？agent 画出预测曲线，环境跑真值，按曲线距离 + 关键事件（是否还 grok、平台期长短）打分。

这个任务的锋利之处：模型手里有 run A 的全部信息，所以**记忆和模式匹配帮不上忙**——它必须做因果推断而非曲线拟合。这也是区分"NCPL 式 surrogate 学到的是相关性还是机制"的一把好尺子，你们自己的 meta-model 都可以拿来当被测选手。

## 三者的定位差异

指纹测的是**生成性理解**（能否构造刺激让机制显形），叠叠乐测的是**结构归因**（知道哪块是承重墙），手术测的是**因果推断**（干预后会发生什么）。三个正好覆盖了 prior 的三个不同侧面，理论上可以做成一个小 suite；但如果只选一个先跑通，我投指纹——它评分最自动、防污染最彻底、而且和你们已有的 Task 3（cherry picking）共享大部分基础设施：cherry picking 是"设计数据集让 A 赢 B"，指纹是它的 N 元推广"设计数据集让所有人互相认得出来"，工程上几乎是免费的升级。

要不要我把指纹任务写成一个可以直接跑的最小原型？用你们现有的 synthetic dataset DSL（Linear Regression / XOR / Modular Addition 那套）做 action space，四个 tiny 架构，lattice 后端，大概能压在几百行以内。