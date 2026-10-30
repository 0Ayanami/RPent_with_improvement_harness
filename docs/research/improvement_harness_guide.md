# Harness2 Memory 构建与评测指南

当前主流程是 **seed 0 构建 memory → 冻结完整语料 → seed 1–10 对照评测**。
每个构建任务执行 k=1 的基线 memory 回合（base）、INV、DOM、composed final 四个回合，随后单独反思。
模型只修改现有 memory 叶子；执行模型、VLA、原语、提示和任务判据保持固定。

方法定义见[设计文档](improvement_harness_design.md)，论文参数依据与差异见
[实验对齐文档](libero_memory_experiment_alignment.md)，模块边界见[迁移记录](improvement_harness_migration.md)。
下文保留单机服务器运行方式；Windows 本地 Codex 与远端 GPU 的配置、SSH 和完整实验命令见
[分布式部署指南](windows_codex_remote_libero.md)。

## 1. 配置和本地语料

正式配置保存在源码目录，未依赖被忽略的 `examples/`：

- [标准 LIBERO](../../robots/libero/config/improvement/libero_memory.yaml)：40 个构建任务，160 个构建 rollout，400 个评测回合/组。
- [LIBERO-Pro](../../robots/libero/config/improvement/libero_pro_memory.yaml)：80 个构建任务，320 个构建 rollout，800 个评测回合/组。
- [Windows Codex 远端执行模板](../../robots/libero/config/improvement/windows_codex_remote.yaml)：标准 LIBERO 矩阵，由 `setup` 生成独立本地配置。

所有路径相对配置文件。先准备：

1. `initial_memory.path`：干净初始语料，至少包含一个 `MEMORY.md`。空构建可以仅写 `# Memory`。
2. `reference_memory.path`：固定的原 Harness-VLA memory；在 `provenance` 填写来源、版本、生成模型和已知历史问题。
3. `execution` 与 `evaluation_execution`：相同 Planner、推理强度、VLA 身份和环境设置，可使用不同 episode budget。
4. `editor`：INV、DOM、Compose 和反思使用的模型及独立请求限制。
5. `vla_identity`：固定 checkpoint revision 和文件 SHA256，并核对实际服务的权重。

模板中的“运行前填写”必须替换后才能构建。静态计划允许保留模板标记，但不验证权重真实性。
初始语料若有 `task-specific/`、`results/` 或已有叶子，快照会保留；系统不会自动清除污染。
来源声明必须反映真实内容。主要研究组应从干净初始语料开始。

准备参考语料可以沿用 `robots.libero.memory.sync_version()`。例如在服务器运行以下 Python，
将返回的目录填写到 `reference_memory.path`，并把下载缓存版本指针 JSON 的 `commit` 记入来源：

```python
from pathlib import Path
from robots.libero.memory import sync_version

root = sync_version(version="GPT_5.5_xhigh", cache_dir=Path("memory/reference_cache"))
print(root)
```

该调用需要网络；`plan`、`construct` 和 `evaluate` 均不会自动下载 HF memory。
历史参考 memory 不能直接赋予论文历史分数；要在当前环境重跑。

## 2. Windows 离线检查

在仓库根目录使用项目虚拟环境：

```powershell
.venv/Scripts/python.exe -m rpent.cli.improvement plan --config robots/libero/config/improvement/libero_memory.yaml
```

命令校验配置和现有 local memory，复制为内容寻址快照，生成 `output_dir/plan.json`。
计划包含构建/评测 case 表、语料 hash、scaffold 指纹和预计 rollout 数，不请求模型、不启动 Env/VLA。
路径不存在时先准备目录；不同计划必须使用新的 `output_dir`。

`suite_names` 可显式选择 benchmark 中的 suite，例如 `suite_names: [libero_object]`；省略时保留完整 suite 表。将它与 `task_ids: [0]`、`evaluation_seeds: [1]` 组合，可只构建一个任务并为每组预声明一个评测回合。`task_ids` 或 `evaluation_seeds` 也可以单独缩小，用于服务器 smoke，但这不再是论文完整矩阵。
`evaluation_seeds` 禁止包含 0。不同 seed 是否代表不同初态仍需服务器检查。

## 3. 服务器构建

按现有 [LIBERO 环境指南](../source-zh/rst_source/simulators/libero.rst)安装对应 extra、
RLinf、模拟器资源、SAM3 和 VLA。设置模型提供方凭据及 `PI05_CHECKPOINT_PATH`，然后从仓库根目录执行：

```bash
python -m rpent.cli.improvement construct --config robots/libero/config/improvement/libero_memory.yaml
```

每个任务按固定顺序：

1. 冻结当前 memory 和以前任务的 meta-library 前缀，执行 base。该回合不使用独立的 `reference_memory` 对照语料。
2. INV、DOM 独立查看相同 base，分别生成多叶子修订，并各执行一次 probe。
3. Compose 查看两份候选、轨迹和说明，返回相对原基线的完整修改集合；组合版本执行 final。
4. k=1 直接采用 final，不按原生成败从 INV/DOM 中选优。模型可以显式返回空修改以保留基线。
5. 单独反思所有修改与执行差异；有受支持的 lesson 才写入独立 meta-library，供后续任务检索。
6. 若选中 final 有真实成功轨迹，导出原生 audit 和 recipe 配对，发布到原有 flat `task-specific/`。

三个编辑阶段各最多两次语义尝试。无效后回退基版本，并执行该阶段原定 probe。
正常任务有四个 rollout、三次编辑调用和一次反思调用；格式修复另计请求。
超预算、清理失败、原生证据不可用或初态不匹配会停止构建。反思服务失败不撤销已采用的 final。
编辑上下文有明确长度限制：优先提供当前任务和已读取的完整叶子，其余只列路径和 hash；
未提供完整源的已有叶子不能被修改。轨迹按完整消息/步骤保留首尾，记录遗漏数量，原始产物完整保存在 trial 目录。
Composer 仍可看到两个候选的全部修改；必要修改自身超出上下文限制时会明确失败并回退。

默认每个 trial 自启 Env，保证 suite/task/seed 匹配。可借用 VLA/SAM3 endpoint；runner 仅关闭自启服务。
外接 Env 必须逐回合匹配条件并支持 `env.get_initial_state_id`；完整任务流通常不宜固定一个 Env endpoint。
`robot_options.libero_type` 的有效值是 `standard`、`pro`、`plus`。

第一版拒绝恢复已有 construction 目录。失败后保留事件、尝试和成本记录；处理问题后改用新的输出目录。
检索在同一实验中优先同 suite、较新的 lesson，并允许复用其他 suite 的经验；不会读取其他实验的库。
成功 audit/recipe 采用只增发布；初始语料中已有同名但内容不同的记录会被拒绝，不会自动覆盖。

## 4. 冻结评测与对照

构建完成后产生 `frozen-reference.json` 和 `frozen-ppc_meta.json`。它们锁定全层 memory、
构建初态 ID、评测 case、执行模型/预算和 scaffold。每组所有测试回合共用一个最终冻结快照。

```bash
python -m rpent.cli.improvement evaluate --frozen logs/libero-memory-ppc-meta/frozen-reference.json --output-dir logs/libero-eval-reference
python -m rpent.cli.improvement evaluate --frozen logs/libero-memory-ppc-meta/frozen-ppc_meta.json --output-dir logs/libero-eval-ppc-meta
python -m rpent.cli.improvement compare --reports logs/libero-eval-reference/evaluation_report.json logs/libero-eval-ppc-meta/evaluation_report.json --output logs/libero-memory-comparison.json
```

评测不调用编辑器、反思或 meta-library，不合并 memory；输出只能写独立评测目录。
每个回合只有 toolkit 初始 reset，测试中不开放 reset。成功来自环境原生 `terminated`，
在预算耗尽后才观察到的成功不计入合格成功数。评分分母是预声明的全部 case。
基础设施异常产生 invalid 报告，不能删掉失败回合后报告成功率。

对照检查两个组的 case、预算、scaffold、初始/reference 语料身份及真实初态 ID。
测试状态不得与 seed 0 或其他测试 case 重复；重复初态内容即使位于不同 catalog index 也拒绝。
三个 seed/任务条件不同的目录不能冒充配对对照。

要运行无跨任务库消融，复制配置，更改 `experiment_id`、`output_dir` 并设置
`use_meta_library: false`。初始/reference 语料与执行配置保持相同；仍执行任务后反思作为记录，
后续任务不读取 lesson。完成后评测 `frozen-ppc_no_meta.json`，可作为第三组加入 `compare`。

## 5. 产物与证据边界

- `plan.json`：静态输入、case 表和版本指纹。
- `snapshots/`：不可变完整 memory，批量编辑和原生任务记录均带来源 hash。
- `construction/<case>/ppc/`：四个 trial、模型尝试、证据、反思、读取审计和成本。
- `construction/<case>/meta_prefix.json`、`construction/meta/`：任务前冻结视图和任务后 lesson，不提供给执行器。
- `construction_report.json`：完成/中断状态、冻结身份、已知成本和 unknown 项。
- `frozen-*.json`：可独立核验的评测输入；移动到其他机器时先整体同步实验目录，再在服务器重新生成计划/构建，不能手改内部路径/hash冒充原实验。
- `evaluation_report.json`：整体与 suite 成功数、逐 case 原生状态和执行成本。

chunk 全长用于预算准入；当前 RPC 不提供实际推进物理步数时，成本保留 null 和已知小计。
构建总成本与冻结部署成本分别报告。对照未约束原流程的构建计算量，不能把效果差异自动归因于算法本身。

当前执行器和编辑器均可选择 `api` 或 `codex`，旧配置默认保持 API；Windows/远端模板选择 Codex。
本地 Codex 复用账号登录，服务器只运行 Env/VLA/SAM3 和原语 worker；API、SDK/MCP 与论文历史文件协议的差异仍需记录。
API 执行器的文件工具只可读快照、本 trial 输出和三个固定 LIBERO 指南，只能写 trial 输出；这是应用层约束。
普通 tool 轨迹中的原生终止信号可被改进模型看到；私有评分字段和最终评测报告不提供给模型。

离线测试证明流程和隔离契约；真实 GPU 运行、论文分数复现和 memory 提升结论仍需服务器实验。
旧 schema 1 的 `check/run/deploy` 保留为单候选兼容入口，不是本指南的 PPC 方法。
