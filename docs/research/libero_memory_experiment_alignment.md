# LIBERO Memory 构建与冻结评测的实验对齐

本文确定如何比较原有探索流程与 Harness2 驱动的 memory 构建流程。两种方法使用相同的执行模型、冻结 VLA、原语和测试预算，只改变 memory 的构建方法。可以在 Linux GPU 服务器上运行完整管线，也可以在 Windows 上运行 Codex、memory 编辑与实验调度，通过 SSH 调用服务器上的 LIBERO、Pi0.5、SAM3 和原语 worker。分布式配置与命令见[操作指南](windows_codex_remote_libero.md)。

这里的对齐范围是 Harness VLA 的任务、参考 seed、测试 seed、观测权限和成功判定。本文没有新增机器人实测结果，也不把历史排行榜成绩视为当前环境的复现实测。

## 1. 来源与版本

核实日期为 2026-10-20。论文引用锁定为 [Harness VLA，arXiv:2607.08448v5](https://arxiv.org/abs/2607.08448v5)，该版本于 2026-09-24 修订；[HTML 正文与附录](https://arxiv.org/html/2607.08448v5)用于核对协议。后续版本的结果和配置应另行记录，不能替换实验启动时的来源。

| 要核对的内容 | 论文位置 | 对本实验的约束 |
| --- | --- | --- |
| 构建与部署两个阶段 | §2.2 | 参考实例可探索；测试时读取已构建的 memory，禁止 reset |
| 固定动作接口与冻结 VLA | §2.3、§3.1、附录 D.1 | 执行原语和 VLA 权重在测试前固定 |
| 标准 LIBERO 的任务和 seed | 附录 C.1、表 10 | 40 个任务，s0 构建，s1–s10 测试，共 400 次 |
| LIBERO-PRO 的任务和 seed | 附录 C.2、表 11 | 80 个任务，s0 构建，s1–s10 测试，共 800 次 |
| 成功判定 | 附录 C 开头 | 使用 benchmark 的二值完成条件 |
| memory 内容和更新 | 附录 A、E.3、E.4 | 成功任务保存 audit 和命令序列，失败保留负证据，空间参数重新定位 |
| 观测权限 | 附录 A、E.2 | 依据 RGB-D、world map 和机器人状态决策，不读取特权物体姿态 |
| GPT-6 Astra 的补充结果 | 附录 H、表 21 | 属于 v5 的补充实验，需同时保留发布语料的历史条件 |

仓库实现与论文分开取证。suite 标识以 [LIBERO 注册配置](../../robots/libero/robot_spec.py)为准；当前推理参数见 [Pi0.5 服务](../../rpent/robots/components/pi05_vla_server.py)。历史复现说明见 [LIBERO 中文指南](../source-zh/rst_source/simulators/libero.rst)，发布语料来源见作者维护的 [LIBERO memory 说明](https://huggingface.co/datasets/RLinf/RPent-memory/blob/main/libero/README.md)。实际运行必须把移动分支解析为完整 commit，并在实验产物中保存；文档链接本身不能替代版本锁。

## 2. 任务矩阵与划分

标准 LIBERO 与 LIBERO-PRO 分开汇总。每个 suite 的 `task_id` 为 0–9；论文中的 Long / LIBERO-10 在代码中使用 `libero_10`。

| 任务家族 | 标准 suite | PRO Task suite | PRO Swap suite |
| --- | --- | --- | --- |
| Spatial | `libero_spatial` | `libero_spatial_task` | `libero_spatial_swap` |
| Object | `libero_object` | `libero_object_task` | `libero_object_swap` |
| Goal | `libero_goal` | `libero_goal_task` | `libero_goal_swap` |
| LIBERO-10 / Long | `libero_10` | `libero_10_task` | `libero_10_swap` |

Task 扰动改变有效目标或任务条件，Swap 扰动在保持指令的情况下交换或重排物体初始位置。这两个设置分别对应论文的 T 和 S。本轮不把 `*_lan`、`*_object` 或 `libero_90` 纳入上述分母。

| 阶段 | 标准 LIBERO | LIBERO-PRO | 可用反馈 |
| --- | --- | --- | --- |
| memory 构建 | 4 suite × 10 task × s0 | 8 suite × 10 task × s0 | 本构建任务的观测、执行轨迹和原生成功信号 |
| 冻结评测 | 40 task × 10 seed = 400 次 | 80 task × 10 seed = 800 次 | 当前 episode 的观测和原生成功信号，仅用于执行和记录 |

正式评测 seed 固定为整数 1–10，构建 seed 固定为 0。s0 的基线回合（base）、候选 probe、组合后 probe 和构建重试都属于构建成本，不计入测试分母。上述设置评估的是相同任务身份上的新初始状态，不是未见任务泛化。

构建任务顺序在实验开始前固定并保存。task 后回顾只向之后的构建任务提供经验，不能读取冻结评测的轨迹、分数或失败案例。完成全部构建任务和回顾后，再冻结整个语料。全部测试 episode 读取同一语料版本；分别构建的标准和 PRO 语料必须使用不同实验标识并分别报告。

### 2.1 核对真实初始状态

不同 seed 不一定对应不同布局。[当前环境选择实现](../../robots/libero/env_server.py)使用 `seed % trials` 选择任务的已存初始状态。运行前必须对每个 `(suite, task_id)` 解析 s0–s10 的真实状态身份，核对：

1. 每个状态身份包含实际 suite、任务、初始状态索引及状态内容摘要。
2. s0 的状态不出现在 s1–s10 中；s1–s10 之间也不重复。
3. 原构建法和新构建法的配对评测使用相同状态身份。
4. 环境服务报告的任务、seed、horizon 与请求一致。

缺失状态身份、状态别名或重复内容都会使对应单元不满足本协议；不能仅更换标签后仍宣称完成 10 个独立测试布局。原始运行记录和未完成单元应保留，解决状态划分后再开展正式比较。

## 3. 固定执行条件

论文使用 RLinf 发布的 `pi05_libero130_fullshot`，即基于 LIBERO-130 的 π0.5-SFT。仓库提供的下载入口为 [RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT](https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT)。构建与评测均保持该 VLA 的权重冻结。

每个实验应保存以下身份信息，原构建法和新构建法逐项相同：

| 项目 | 需要固定和记录的内容 |
| --- | --- |
| 执行规划器 | backend、模型完整标识、服务版本或可用的版本标识、reasoning effort、token 上限 |
| VLA | checkpoint 仓库、完整 revision、实际加载文件的 SHA-256、归一化统计文件及其摘要 |
| VLA 推理 | embodiment、配置名、action chunk、推理步数、输入图像、精度和运行时版本 |
| 感知 | SAM3 checkpoint、服务配置与相机/深度接口版本 |
| 环境 | RPent、RLinf、LIBERO 或 LIBERO-PRO 的代码版本、任务资源、仿真配置与初始状态身份 |
| 动作接口 | 可用原语、原语实现、执行阈值与启停条件、是否暴露直接动作接口 |
| 预算 | 每 episode 的规划轮数、墙钟、环境步数、VLA 调用限制，以及构建阶段的总预算 |
| memory | 起始语料、最终语料、每个发布文件及整体快照摘要 |

`vla_identity` 等配置字段是对实际服务的声明。填写一个名称不能证明服务真的加载了相应权重；需与服务启动记录及文件摘要一起归档。

当前 Pi0.5 LIBERO preset 设置 `config_name=pi05_libero`、`action_chunk=5`、`num_steps=5`、两路图像与 7 维环境 action。它们是当前代码配置，不是已经从论文核实的全部历史推理参数。新实验应固定这些值，不能由 memory 候选修改服务或原语源码。

## 4. Memory 构建与评测权限

新构建法使用 Harness2 的 task-level k=1 propose–probe–compose，并在每个构建任务后做 harness-level 回顾；详细流程见 [研究设计](improvement_harness_design.md)。新算法编辑 RPent 现有的 `global/`、`task-family/` 和索引，真实成功任务的 audit/recipe 从执行产物发布。论文明确任务 memory 和 global memory；`task-family/` 是 RPent 现有存储层，不应表述为论文独立提出的第三种 memory。

原构建法和新构建法应从相同起始语料开始。若起始语料为空，报告从头构建；若继承作者发布语料，记录其精确 revision，并报告为继承语料上的改进。两者不能合并为同一个实验条件。

当前运行入口接收 `initial_memory` 作为新流程的起点，接收已经准备好的 `reference_memory` 作为冻结对照，不会代替操作者运行一遍原探索流程。若参考语料来自作者历史发布，本轮直接比较的是两份语料的冻结表现。要归因于构建算法本身，还需从相同起点按原探索方法生成参考语料，并另行控制、记录构建预算。

| 权限 | 构建阶段 | 冻结评测阶段 |
| --- | --- | --- |
| 获取 RGB-D、深度投影 world map、机器人自身状态 | 允许 | 允许 |
| 获取当前任务指令、可见物体名和原生成功信号 | 允许 | 允许 |
| 读取 BDDL 中的真值坐标、物体真值姿态或 simulator internals 来决策 | 禁止 | 禁止 |
| 同 episode 内重新定位、重新 staging、再次调用 VLA | 预算内允许 | 预算内允许 |
| reset 后启动新的尝试 | 构建预算内允许 | 禁止 |
| 更新 memory、任务经验与回顾库 | 仅使用构建证据 | 禁止 |
| 更改 VLA 权重、原语代码或基础执行 prompt | 不属于本轮 memory 构建实验 | 禁止 |

完整冻结快照包括 `MEMORY.md` 和所有可读取层。评测产生的 audit、视频、轨迹及统计写入输出目录，不能合并回语料。所有测试 episode 结束后再次核对语料摘要；文件内容变化说明运行偏离冻结协议。

使用环境原生成功信号为候选提供反馈，是本项目面向机器人的实验选择。应标注为 native-feedback-assisted 的 Harness2 流程迁移，不能把该反馈条件描述成 Harness2 原实验已验证的机器人设置。

## 5. 预算的证据边界

论文 §2.2 只定性说明探索阶段允许 reset、墙钟预算更宽裕，测试阶段禁止 reset 并缩短执行步数。本文未从 v5 核实到完整的数值预算、主表每个规划器的推理强度、temperature、采样配置或 checkpoint 文件摘要。这些项目必须由新实验显式固定，并作为配置差异披露。

| 参数或配置 | 可核对的值 | 证据及适用范围 |
| --- | --- | --- |
| 历史复现分支默认规划轮数 | 100 | [reproduce/libero README](https://github.com/RLinf/RPent/tree/reproduce/libero)，代码默认值，不证明论文全部运行采用该值 |
| 历史复现分支默认 reply token 上限 | 8192 | 同上 |
| 历史复现分支默认环境步数上限 | 10000 | 同上 |
| 仓库 Long 复现命令 | `gpt-5.5`、`xhigh`、100 turns、5000 秒、10000 env steps | [中文 LIBERO 指南的实验复现段](../source-zh/rst_source/simulators/libero.rst)，属于仓库复现配置 |
| 当前普通探索配置 | 最多 3 个 session，每 session 最多 5 次尝试 | [LIBERO 配置](../../robots/libero/robot_spec.py)和[记忆指南](../source-zh/rst_source/guides/memory.rst)，不是已核实的论文探索预算 |
| Astra 补充实验 | Codex、GPT-6 Astra、low reasoning effort | 论文附录 H；仍需保留具体代码和语料版本 |
| 公开 GPT-5.5 语料生成配置 | Codex、GPT-5.5、xhigh | [作者语料说明](https://huggingface.co/datasets/RLinf/RPent-memory/blob/main/libero/README.md)；语料生成 effort 不会自动设置新运行的模型或 effort |

k=1 的正常 task-level 流程包含 base、INV probe、DOM probe 和组合后 final probe，不能把它记作一次物理 episode。base 使用任务开始时的 memory，不是独立的冻结 reference 对照语料。比较构建成本时记录实际 rollout、reset、环境步数、VLA 调用、墙钟和模型用量；缺失计数记为未知，不补成零。

## 6. 历史结果与可比性

历史成绩可以用于说明研究背景。新实验的成败由同一当前环境下重跑的对照决定。

| 历史结果 | 条件和限制 |
| --- | --- |
| 标准 LIBERO，Harness VLA / Claude Code 96.0% | v5 表 2，384/400；不等于新 memory 语料已取得这一结果 |
| LIBERO-PRO，GPT-5.5 72.1% | v5 表 3/21；Long Task/Swap 为 52%/49% |
| LIBERO-PRO，GPT-5.5 75.13% | 仓库排行榜将六项论文成绩与后续 Long 70%/55% 合并，不是 v5 主表原值 |
| LIBERO-PRO，Opus-4.7 82.4% | v5 表 3/21；需与对应模型、语料和预算一起引用 |
| LIBERO-PRO，GPT-6 Astra 92.63% | v5 附录 H 表 21，741/800；历史运行使用两个分别冻结的语料快照 |

作者 [Astra history](https://huggingface.co/datasets/RLinf/RPent-memory/blob/main/libero/README.md)明确说明：Long 与其他六个 suite 使用各自的原始冻结快照，合并后的发布语料未重新评估；memory 生成于 runtime commit `014a0fa`，早于 LIBERO-PRO scene-seed fix。这些历史结果不是合并语料或修复后环境的新增评测。因此不能把 92.63% 当作“同一个冻结语料在当前真实 heldout 布局上的基线”。这一限制也不单凭历史说明就证明每一条旧轨迹都发生了布局重合；具体重合须查旧代码和状态记录。

| 历史资源 | 精确 revision |
| --- | --- |
| GPT-5.5 迁移前语料 | `21a62795fe3b7e500c8381ac47938f6d713ebe18` |
| Astra Long 原始快照 | `cf5d14ce9b3ec6c72de5477ec0fea806a3884efa` |
| Astra Spatial/Object/Goal 原始快照 | `984c57f7c6caf48b572f5926851dd9134ae041d8` |

语料目录迁移、重建索引或合并两个快照，不构成新的环境评测。复用历史语料必须保留来源摘要和当时的适用范围。

## 7. 结果记录与报告

每个评测 episode 至少保留 `(suite, task_id, seed, initial_state_id)`、执行配置摘要、memory 快照摘要、原生成功值、结束原因及实际成本。成功只由环境完成条件确定，`finish`、抓取工具的局部成功或视频中看起来完成都不能替代该值。

标准和 PRO 分别给出每 suite 的成功次数/100，以及总成功次数/400 或 /800。只有完整覆盖既定矩阵才能使用完整分母。环境或服务错误、超时、主动结束、缺失运行和有效任务失败分别记录；不能静默丢弃失败运行后缩小分母，也不能把尚未执行的项目填成实测失败。重跑必须保留原记录和预先规定的重跑规则，不能逐 episode 选择最好的结果。

原构建法与新构建法按相同任务和真实初始状态配对比较，并分别报告构建成本与冻结测试成本。早期小范围运行用于检查管线和资源，不足以替代完整基准。离线单元测试、配置检查和摘要校验也不构成 VLA 或 benchmark 成功率证据。

## 8. 运行入口与产物

命令入口为 `python -m rpent.cli.improvement`。本节保留单机服务器示例，使用仓库路径 `/work/RPent` 和标准 LIBERO 的 API 配置；仿真、VLA、SAM3 及模型凭据先按 [LIBERO 安装与资源说明](../source-zh/rst_source/simulators/libero.rst)准备。不同 LIBERO 变体使用各自的 Python 环境。Windows 本地 Codex 方式使用独立配置，按[分布式部署指南](windows_codex_remote_libero.md)在本地执行同样的构建和评测入口；Codex 登录仅在 Windows 侧需要。

### 8.1 准备配置

标准实验配置为 [libero_memory.yaml](../../robots/libero/config/improvement/libero_memory.yaml)，PRO 配置为 [libero_pro_memory.yaml](../../robots/libero/config/improvement/libero_pro_memory.yaml)。路径相对于配置文件解析。正式运行前填写真实的模型、checkpoint 身份、初始语料、参考语料、来源说明及新的输出目录；模板中的“运行前填写”等占位内容不能作为来源证据。

从空语料开始时，初始目录至少包含 `MEMORY.md`，不混入旧的 global、task-specific 或 results。参考目录指向一份已经下载、核对并锁定版本的完整语料，来源中说明对应 benchmark。标准和 PRO 的参考语料适用范围需要分别核对，不能因为格式相同就把 PRO 任务经验当作标准任务的原始参考。

完整配置中标准实验预期构建 40 个任务，每任务四次 rollout，共 160 次；冻结测试每组 400 次。PRO 预期构建 80 个任务，共 320 次；冻结测试每组 800 次。任务未完成或配置裁剪后，应报告实际覆盖，不能沿用完整分母。

### 8.2 离线检查与构建

先生成本地计划，检查输入快照、任务表和配置。它不会启动模型服务或仿真，也不会证明原生初始状态已经通过检查。

```bash
cd /work/RPent
python -m rpent.cli.improvement plan \
  --config /work/RPent/robots/libero/config/improvement/libero_memory.yaml
```

输出目录中生成 `plan.json` 和输入快照。确认配置与语料后，在 GPU 服务器执行构建：

```bash
python -m rpent.cli.improvement construct \
  --config /work/RPent/robots/libero/config/improvement/libero_memory.yaml
```

构建输出逐任务的 PPC 记录、被冻结的先前回顾视图、模型编辑及反思证据。成功结束后生成 `construction_report.json`、`frozen-reference.json` 和 `frozen-ppc_meta.json`。最后一个文件固定全体测试读取的语料和执行配置。中断或未完成任务保留记录，需要新的输出目录重新开始；当前入口不自动续跑。

将 `use_meta_library` 设为 `false`，并使用不同的实验标识和输出目录，可以生成不检索先前 task 回顾的对照；其冻结文件名为 `frozen-ppc_no_meta.json`。该模式仍记录每个任务的反思，仅不向后续任务检索。两组应使用相同任务顺序、起始语料及执行配置。

### 8.3 冻结评测与配对汇总

以下路径对应模板中的标准实验输出。若修改了 `output_dir`，同时调整全部产物路径。每次 `evaluate` 都要求新的评测输出目录。

```bash
python -m rpent.cli.improvement evaluate \
  --frozen /work/RPent/logs/libero-memory-ppc-meta/frozen-reference.json \
  --output-dir /work/RPent/logs/libero-memory-eval-reference

python -m rpent.cli.improvement evaluate \
  --frozen /work/RPent/logs/libero-memory-ppc-meta/frozen-ppc_meta.json \
  --output-dir /work/RPent/logs/libero-memory-eval-ppc-meta

python -m rpent.cli.improvement compare \
  --reports \
    /work/RPent/logs/libero-memory-eval-reference/evaluation_report.json \
    /work/RPent/logs/libero-memory-eval-ppc-meta/evaluation_report.json \
  --output /work/RPent/logs/libero-memory-comparison.json
```

`evaluate` 仅使用冻结文件、对应语料和执行配置，不运行编辑器，也不检索构建回顾库。输出包含逐 episode 记录及 `evaluation_report.json`。语料被修改、初始状态与构建重合、测试布局重复、原生成功值缺失或清理失败，均不能形成有效的完整评测。`compare` 仅接受执行条件一致、任务表相同且真实初始状态匹配的完整组，输出各组成功率及相对 reference 的百分点差。

PRO 使用其配置文件和 `libero-pro-memory-ppc-meta` 输出目录，其余流程相同。小样本先导运行通过配置裁剪 `task_ids` 和 `evaluation_seeds`；它只用于验证流程，结果不标为 400/800 次正式实验。上面的命令是服务器运行步骤，本文没有声称它们已完成真实模型或仿真评测。
