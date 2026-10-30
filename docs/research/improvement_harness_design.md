# 基于 Harness2 的 Memory 构建与 LIBERO 实验设计

- 更新日期：2026-10-26。
- 范围：迁移 Harness2 的双层递归流程，编辑对象限定为 RPent 原有 memory。
- 状态：已按本设计完成代码迁移、memory 适配和离线验证；服务器运行与性能增益尚未验证。
- 文档仅保存在 `docs/research`，使用中文。

## 1. 开发目标

第一版研究问题是：在相同 Harness-VLA 执行模型、VLA checkpoint、工具、任务和测试预算下，
由 Harness2 的流程构建的 memory，是否比原 memory 构建方式带来更好的冻结部署表现？

改进对象不是 VLA 权重、八组件的整个 harness、原语代码或基础 system prompt。
保留现有 memory 的格式、目录和读取方式，将模型生成的文件修改编译为经过校验的语料版本。
首先完成构建方法与实验管线，不同时加入规划调度、动作粒度或模型训练优化。

## 2. 两层递归与 k=1

参考 [Harness2 论文](https://arxiv.org/html/2610.14224v1)及本地官方源码
`harness2/harness2/core/`（版本 `1cfe254e8d25b15f37048c3730134686abe60335`）。

一次 k=1 的 task-level recursion 包含：

1. 用本任务开始时的 memory 执行一次基线回合（base）。
2. INV proposer 从同一 base 提出可跨任务复用的 memory 修改。
3. DOM proposer 从同一 base 提出面向当前任务/扰动域的 memory 修改。
4. 两个候选分别从同一基版本物化并执行 probe。
5. Composer 阅读 base、两个候选的修改和执行证据，生成一个新的组合版本；
   可以保留基线，也可以选取或重新组织候选内容，不能用简单字符串拼接替代组合。
6. 组合版本自身执行一次 final probe。k=1 只有一个组合结果，不增加多候选 selector。

正常路径每任务四次完整 rollout：base、INV、DOM、composed final。
一次 PPC 包含三次编辑阶段，不意味着只执行一次任务。物理环境试验串行执行，候选上下文独立。
这里的 base 是当前任务开始时的 memory 回合，与冻结评测的 `reference_memory` 对照组不同。

任务结束后执行 harness-level recursion：模型回顾全部修改、轨迹及组合决策，输出带证据引用、
适用范围、反例和不确定性的改进经验。冻结该任务的历史经验视图，反思只对之后的构建任务可见。
反思服务失败或返回空经验时保留完整尝试记录，不能用模板化的“成功”文本冒充模型回顾。

k=1 的性能是原论文在其 benchmark 上的结果，不是本项目已经取得的机器人实验结论。

## 3. Memory 编辑面

继续使用现有目录：

| 层 | 内容与更新方式 |
| --- | --- |
| `global/` | 可复用的原语、感知、策略、失败和基础设施规则；INV 主要编辑这一层 |
| `task-family/` | 当前 suite/regime/task 的操作策略；DOM 主要编辑这一层 |
| `task-specific/` | 真实成功 episode 的 audit JSON 与 recipe JSONL；从执行产物发布，不让模型伪造轨迹 |
| `MEMORY.md` | 现有索引；更新时保留无关说明，纳入整体 hash |

一个候选可以修改多个叶子。操作沿用 add/revise/narrow_scope/retire，修订保留 ID，
每个已有叶子检查旧内容 hash，拒绝重复目标、越界路径、错误 scope 和不符合 schema 的内容。
INV、DOM 都相对同一基版本；Composer 返回相对该基版本的完整修改集合。
采用版本可成为下一个构建任务的基版本，不再限制“所有任务只能对固定 E0 改一个叶子”。

task-specific 的成功记录与最终采用的执行轨迹绑定，保留来源 task/seed、trial ID 和 hash。
没有真实成功轨迹时不发布成功 audit/recipe。原始轨迹和改进经验均不直接混入执行 prompt。

## 4. 构建阶段与测试阶段

构建阶段允许 seed 0 的任务交互、PPC 和任务后反思。所有 trial、编辑请求、失败、回退、
reset 和反思都计入构建成本。初始语料与历史经验来源显式声明；干净构建从空语料开始，
不能默认下载已经含目标测试经验的 HF 语料。

构建结束冻结完整 memory，生成内容 hash、case 划分、模型/checkpoint/config 指纹和来源 manifest。
随后所有测试 episode 读取同一冻结语料；测试不调用 proposer/composer/reflection，不写 memory，
不以测试成功率选择候选，不更新跨任务库。测试结果只进入独立评测报告。

原生环境终止与工具反馈可能已出现在构建轨迹中；本适配明确记录这种反馈权限。
不能声称它严格复现办公 benchmark 的无外部评分反馈设定。独立测试集的结果不向编辑器暴露。

## 5. 原论文实验对齐

以 [Harness-VLA v5](https://arxiv.org/html/2607.08448v5) 为固定参考，区分论文主表、
仓库后续 leaderboard 和不同 memory 发布版本，不能混用它们声称复现同一配置。

已核对的任务/seed 协议：

| 实验 | 构建 | 冻结测试 |
| --- | --- | --- |
| Standard LIBERO | Spatial/Object/Goal/LIBERO-10；每 suite 10 task 的 seed 0 | 每 task seed 1–10，共 400 episode |
| LIBERO-Pro 扩展 | 四类任务的 Task/Swap，共 8 suite；各 task seed 0 | 每 task seed 1–10，共 800 episode |

上述是同任务未见布局评估；若要考察未见任务，另行预声明 disjoint task split，不与论文 few-shot 协议混称。
冻结 VLA 为论文的 `pi05_libero130_fullshot`，checkpoint 具体来源与字节 hash 在服务器运行前填写。
Planner 模型、推理强度、工具、环境版本和每 episode 预算在比较组中保持一致。
论文未公开的参数标为未公开；当前 RPent 示例中的默认值标为实现选择，不能冒充论文值。

执行后端与原论文存在差异时，报告为“任务与 seed 协议对齐的受控 memory 比较”。
当前支持 API Planner 和 Codex SDK，后者可在 Windows 本地规划、通过 SSH 调用 Linux 原语 worker。
API 工具调用、Codex SDK/MCP 与历史 coding-agent/file-mediated backend 的差别必须记录，
不得把不同后端/模型/已有 memory 的绝对分数直接解释为 memory 方法优劣。

## 6. 比较组

| 组 | 用途 |
| --- | --- |
| 原 Harness-VLA memory / 冻结参考语料 | 相同任务、执行模型、预算下的主要对照；要求记录语料来源与 hash |
| PPC memory，关闭跨任务改进库 | 检查任务内 propose–probe–compose 的作用 |
| PPC memory，开启任务后反思与改进库 | 完整双层递归适配 |

可增加同构建交互预算的原流程对照，区分方法收益与更多构建计算；未实现或未运行的组不出结果。
不要求在 Windows 得出任务成功率，只交付可配置的构建、冻结、评测和汇总入口。
Windows 可以运行 Codex、编辑器和实验调度，仿真与 GPU 模型在服务器执行；
部署配置和操作命令见 [Windows 与远端 LIBERO 指南](windows_codex_remote_libero.md)。

## 7. 实现职责

- `rpent/memory/snapshots.py`：通用多叶子事务与不可变快照，不依赖 improvement 领域。
- `rpent/improvement`：阶段模型、PPC、反思、历史库、构建流、冻结实验与结果汇总。
- `robots/libero/improvement.py`：原生初态/成功、Env/VLA 预算入口、真实 audit/recipe 及清理。
- `robots/libero/remote.py`、`remote_worker.py`：本地 Codex 与远端原语执行、快照/产物传输及断连回收。
- `rpent/improvement/ssh.py`：Windows 配置准备、SSH 隧道、worker 检查与真实 smoke 入口。
- `rpent/cli/improvement.py`：实验准备、memory 构建、冻结评测和报告命令。
- `docs/research`：中文设计、使用说明、论文对齐依据和迁移记录。

保留旧单候选配置的显式兼容入口；它不代表 k=1 PPC，也不用于新的默认实验。
不复制完整 CLI 主程序，不把上层改进器注册成递归调用执行器的普通工具。

## 8. 记录、预算与恢复边界

每个 trial 使用独立上下文/输出，运行前后校验语料；先排空工具再读取最终状态并释放自有服务。
外接服务不由 trial 关闭。记录每个阶段的模型请求、token、耗时、执行结果、读取和资源限制。
chunk 按全长做预算准入；实际物理推进数量缺少可靠数据时保留 unknown，不伪记零成本。

正常 k=1 预留四个 rollout 和三次编辑及一次反思的预算。每次格式修复也消耗请求，
超预算和清理失败显式停止，不用“目录存在”表示完成。
第一版可以拒绝恢复中断的构建，但必须保存 interrupted 和已经发生/未知的成本，不能伪造完成结果。
冻结测试恢复若实现，只能复用输入 hash、case、snapshot、结果及产物全部匹配的 finalized 记录。

## 9. 离线验收与服务器交接

离线测试至少覆盖：INV/DOM 共享基线与独立上下文、scope 白名单、多叶子修订不修改源、
Composer 确实消费两份候选证据、composed final 自身执行、任务后反思可影响下一个任务、
任务内库冻结、无效阶段有限重试与 fallback、四 rollout 全计费、seed 0/seed 1–10 隔离、
冻结评测不调用改进模型/不写 memory、真实 task-specific 来源和失败不伪造成功轨迹。

完整单元测试与 lint 按 CONTRIBUTING 执行；Windows 不可用的 POSIX、Torch、GPU/模拟器路径
分别列出。服务器运行后报告整任务成功率、suite/任务分解、构建成本、冻结部署成本、
无效 trial 和 unknown 计数。实现流程正确、GPU smoke 通过和 memory 带来性能提升是三种不同结论。
