# Harness2 Memory 迁移与代码审查索引

更新日期：2026-10-30。本文件集中标注 improvement 相关源码、入口、职责和审查顺序，
核对的实现基线为 `6a22efeebf3a6fe2a96836572c936642f585031a`。
当前方法见[设计](improvement_harness_design.md)、
[使用指南](improvement_harness_guide.md)和[实验对齐依据](libero_memory_experiment_alignment.md)。
Windows 本地 Codex 与 Linux GPU 服务器的部署见[操作指南](windows_codex_remote_libero.md)。
研究文档统一保存在 `docs/research`，只维护中文 Markdown。

## 来源与迁移范围

RPent 开始迁移时基线为 `ef70215467cf3acc7f71e9fc89914d8e707d338d`，
分支为 `codex/improvement-harness-v1`。本地独立项目 `harness2/` 的参考版本为
`1cfe254e8d25b15f37048c3730134686abe60335`，未改动其源码。

迁移保留 Harness2 的 k=1 INV/DOM 提案、独立 probe、Compose、final rollout 与任务后反思，
编辑对象缩小为 RPent 现有 memory。未迁入办公任务、OpenCode substrate、安装器和八组件任意修改面。
不需要安装 harness2 包；执行仍复用 RPent 的 Planner、工具、Env/VLA、RobotSpec 和运行时。

## 建议审查顺序

1. 读下文的调用流程，以及 `stream.py` 的 `StreamConfig`、`models.py` 的 `ExecutionConfig` 和 `TrialResult`，确认构建与评测的输入、输出。
2. 读 `rpent/cli/improvement.py::_stream_command()` 与 `stream.py::construct()`，确认哪个入口会真正启动优化。
3. 读 `ppc.py::PPCController.run()` 与 `ppc_proposer.py`，审查 INV、DOM、Compose、反思的上下文和采用规则。
4. 读 `rpent/memory/snapshots.py::SnapshotStore`，审查模型允许修改什么、如何保留原 memory 架构。
5. 读 `robots/libero/improvement.py::LiberoTrialRunner`，再读远端桥接，核对预算、原生成功、初态与资源清理。
6. 回到 `stream.py::evaluate()`、`compare()` 和对应测试，审查冻结语料和实验对照。最后再看旧 schema 1 兼容模块。

## 主线调用流程

`rpent-improve` 是 [pyproject.toml](../../pyproject.toml) 注册的命令，等价于
`python -m rpent.cli.improvement`。当前 schema 2 主线为：

```text
plan / construct
  └─ load_stream_config → plan：预声明 task/seed，导入 initial/reference 快照
       └─ construct：按 seed 0 任务顺序构建，冻结以前任务的 meta 前缀
            └─ PPCController.run（每个任务 k=1）
                 base rollout
                 → INV、DOM 分别基于同一 base 提案
                 → INV probe、DOM probe（环境回合串行）
                 → Compose → final rollout → 单独反思
            └─ 采用 composed final；成功时发布原生 task-specific 记录
            └─ 累积 memory；有合格 lesson 时加入 meta 前缀
       └─ 冻结 reference 与 ppc_meta / ppc_no_meta 两组评测输入
evaluate：只读冻结 memory，在 seed 1–10 执行普通回合
compare：核对配对条件和原生初态，再计算相对 reference 的差值
```

一次 k=1 构建任务通常包含四个独立 rollout。它不是在单个 Env 回合中修改一次文件后续跑；
同一任务的四个 rollout 必须解析到相同原生初态。INV 与 DOM 的提案相互独立，Compose 相对
任务开始时的 base 返回完整修改集合。正常完成时直接采用唯一 composed final，不按原生成功率
在候选之间选优；无效编辑有有限重试与保留 base 的回退。采用不代表修改已被证明有效。

## 优化主线与数据契约

| 代码位置 | 主要入口 | 作用与审查重点 |
| --- | --- | --- |
| [rpent/cli/improvement.py](../../rpent/cli/improvement.py) | `main()`、`_stream_command()` | 区分 schema 2 的 `plan/construct/evaluate/compare` 与旧 `check/run/deploy`；构造 runner、editor 和 snapshot store。 |
| [rpent/improvement/stream.py](../../rpent/improvement/stream.py) | `StreamConfig`、`FrozenMemory`、`plan()`、`construct()`、`evaluate()`、`compare()` | 管理任务矩阵、顺序构建、跨任务 lesson 前缀、原生记录发布、完整语料冻结和配对评测。`use_meta_library: false` 禁用后续任务对 lesson 的检索，仍记录任务后反思。 |
| [rpent/improvement/ppc.py](../../rpent/improvement/ppc.py) | `PPCEditor`、`PPCController.run()`、`model_evidence()`、`memory_sources()` | 调度单任务四个 rollout、三阶段编辑和反思，检查初态/快照/scaffold 不变性、有限重试、证据引用和失败停止。只将普通执行轨迹及读取审计投影给编辑器。 |
| [rpent/improvement/ppc_models.py](../../rpent/improvement/ppc_models.py) | `MemoryRevision`、`PPCReflection`、`PPCLesson`、`PPCTaskResult` | 定义多叶子修改、可检验假设、预期效果与反证、逐叶子反思、可空 lesson、分阶段成本和任务结果；拒绝重复修改目标和缺少依据的非空修订。 |
| [rpent/improvement/ppc_proposer.py](../../rpent/improvement/ppc_proposer.py) | `APIEditor`、`CodexEditor`、`build_editor()`、`_memory_payload()` | 实现 INV、DOM、Compose 和反思的独立结构化请求及角色提示；限制上下文长度，优先提供已读取叶子的完整源，未完整提供的已有叶子不能编辑。编辑器没有工具权限。 |
| [rpent/improvement/models.py](../../rpent/improvement/models.py) | `TaskCase`、`TrialBudget`、`ExecutionConfig`、`TrialSpec`、`TrialResult`、`TrialCost` | 共享回合、模型、远端、预算、原生成功、初态、读取审计和成本契约；同时包含旧单候选协议模型，不能仅按文件名判断是否属于旧实现。 |
| [rpent/improvement/__init__.py](../../rpent/improvement/__init__.py) | 包入口 | 仅声明模块用途，没有自动启动优化或替换普通执行器的副作用。 |

## 现有 memory 的扩展位置

原有 memory 叶子 schema、三层目录与读取工具继续使用。多叶子事务生成 v2 snapshot manifest；
旧 v1 和单叶子 API 兼容。模型不能编辑 task-specific，成功记录由环境终止与记录命令生成。
meta-library 与执行 corpus 分离，任务内只读以前任务的冻结前缀，评测不读取它。

| 代码位置 | 主要入口 | 作用与审查重点 |
| --- | --- | --- |
| [rpent/memory/snapshots.py](../../rpent/memory/snapshots.py) | `MemoryEdit`、`MemorySnapshot`、`SnapshotStore.capture/apply_many/verify/publish_task_records` | 在现有 corpus 外保存 manifest，复制完整语料产生内容寻址快照；原子应用 `add/revise/narrow_scope/retire`，校验旧文件 hash、路径与 schema。模型编辑范围为 `global/*.md` 和 `task-family/*.md`；原生 audit/recipe 走单独的可信发布接口。 |
| [rpent/memory/manager.py](../../rpent/memory/manager.py) | `MemoryManager.__init__()`、`record_read()`、`_authorize()` | 沿用现有 memory 管理器，增加回合读写目录约束和读取审计，区分源字节 hash 与实际提供内容 hash/截断；UTF-8 读写，POSIX 锁仅在原有合并操作处导入。 |
| [rpent/tools/common.py](../../rpent/tools/common.py) | `CommonTools.read_text_file()` | 在实际文件工具读取时记录源字节与返回文本，避免文件变化或截断被误当成完整读取。 |

`MEMORY.md` 索引由快照存储按叶子变更维护，模型不能借此直接修改提示、工具或 task-specific
记录。初始 corpus 中已有的其他层和记录会被保留，不能把“快照已捕获”当成“语料无污染”。
新主线的 meta lesson 保存在 `construction/meta/` 及冻结前缀中，不使用旧 `ExperienceLibrary`
向执行器注入经验。

## LIBERO 执行与共享代码改动

这些代码承载一次回合的执行和证据采集，也被普通 baseline 复用。调用 runner 本身不会调用编辑器。

| 代码位置 | 主要入口 | 作用与审查重点 |
| --- | --- | --- |
| [robots/libero/improvement.py](../../robots/libero/improvement.py) | `LiberoTrialRunner.run()`、`build_trial_runner()`、`_BudgetEnv`、`_BudgetModel` | 使用原有 RobotSpec、Planner、Toolkit、Env/VLA 执行冻结 memory；每回合一次初始 reset，限制时间/动作准入/VLA 次数，读取原生 `terminated`，导出成功 audit/recipe，只停止自身启动的服务。按 `execution.remote` 选择本地或远端 runner。 |
| [robots/libero/env_server.py](../../robots/libero/env_server.py) | `make_env()`、`LiberoEnvFacade.get_initial_state_id()` | 根据实际选中的初态数组生成含 SHA256 的身份，并通过 RPC 暴露；允许将仿真视频目录置于项目工作区。 |
| [robots/libero/env_client.py](../../robots/libero/env_client.py) | `LiberoEnvClient.__init__()`、`get_initial_state_id()` | 获取原生初态身份；允许由 toolkit 接管初始 reset，避免连接和回合初始化各重置一次。 |
| [robots/libero/robot_spec.py](../../robots/libero/robot_spec.py) | `get_toolkit()`、`_init_runtime()` | 注入受审计且只读的 `MemoryManager`，传递 reset 所有权设置，复用原机器人注册和运行时。 |
| [robots/libero/memory.py](../../robots/libero/memory.py) | `sync_version()` | 将 POSIX `fcntl` 导入移到服务器同步操作处，使 Windows 本地语料与机器人发现可导入；不是新的优化器。 |
| [rpent/planner/base.py](../../rpent/planner/base.py)、[rpent/planner/codex.py](../../rpent/planner/codex.py) | `build_planner()`、`CodexPlanner`、`build_codex_config()`、`validate_codex_tools_event()` | 增加显式 `tools_only` 选项；实验执行器只使用 RPent MCP 工具，编辑器不接入工具，并检查 SDK 事件。普通调用默认值保持原有设置。 |
| [rpent/utils/serialization.py](../../rpent/utils/serialization.py)、[rpent/cli/main.py](../../rpent/cli/main.py) | `strip_inline_images()`、`_serialize_messages()` | 复用去除内联图像的轨迹序列化，避免把图像大载荷写入编辑上下文；不改变任务优化规则。 |

## Windows/Codex 远端基础设施与 baseline

| 代码位置 | 主要入口 | 作用与审查重点 |
| --- | --- | --- |
| [robots/libero/remote.py](../../robots/libero/remote.py) | `RemoteLiberoTrialRunner`、`RemoteToolkit`、`source_identity()`、`remote_tool_arguments()` | 本地创建 Codex 会话，向服务器传送快照、转发普通工具调用和收回产物；校验协议/源码摘要/传输 hash，转换 Windows 与 Linux 文件路径，维持租约并在异常时取消远端回合。 |
| [robots/libero/remote_worker.py](../../robots/libero/remote_worker.py) | `create_app()`、`_Session`、`_Worker` | 服务器单活动 trial，校验快照与路径；调用现有 `LiberoTrialRunner`，执行原语/VLA，回传原生结果；以调用序号约束重放，断连租约到期或退出时清理自有回合。 |
| [rpent/improvement/ssh.py](../../rpent/improvement/ssh.py) | `SSHConnectionConfig`、`setup()`、`tunnel_command()`、`check()`、`smoke()` | 生成本地实验配置，构造回环 SSH 隧道，检查 worker 身份和空闲状态；内置 smoke 是独立的小预算连接检查。 |
| [robots/libero/baseline_smoke.py](../../robots/libero/baseline_smoke.py) | `run()`、`smoke_report()` | 用固定原始 memory 执行普通 Harness-VLA 单回合，检查真实 Pi0.5 调用、观测、动作、finish 和清理；不调用 PPC、编辑、反思、采用或 meta 检索。报告分别记录链路是否通过与任务原生是否成功。 |
| [robots/libero/deployment/](../../robots/libero/deployment/) | `environment.sh`、`manage_services.py`、`fetch_artifacts.py`、`run-smoke.ps1` | 固定工作区环境、管理本项目模型服务、下载校验资源和启动 baseline；`resources.json`、`memory-manifest.json`、`requirements-server.txt` 固定部署输入。这里的 deployment 与下文旧资格选择模块用途不同。 |

baseline 虽复用 `rpent.improvement.models`、快照和 trial 生命周期，优化算法入口并未执行。
不能因为源码路径含 improvement 就把 baseline 冒烟视为 PPC 实验。

## 与旧 MVP 的关系

此前的单叶子 patch → 原生成功改善门槛 → confirmation/regression 流程，
保留在 `controller.py`、`proposer.py`、`library.py` 和 `deployment.py`，仅供旧 schema 1 配置显式调用。
它的 accepted/rejected 报告不作为 PPC 的 final 选择器，旧测试结果也不作为新流程已经验证的证据。

新流程不使用 held-out seed 为提案或 Compose 选优；k=1 的 composed final 被采用不表示它已证明有效。
Native terminal 可以出现在普通工具观测中；私有 TrialResult 成功字段、资格报告和最终评测不进入改进上下文。

| 代码位置 | 主要入口 | 作用与主线关系 |
| --- | --- | --- |
| [controller.py](../../rpent/improvement/controller.py) | `ImprovementHarness.run()` | 旧单候选资格门槛、confirmation/regression 调度。主线仍复用本文件中的 `TrialRunner`、`scaffold_fingerprint()` 和成本汇总，不使用旧接受/拒绝选择器。 |
| [proposer.py](../../rpent/improvement/proposer.py) | `APIProposer`、`manual_proposal()` | 旧单叶子 API/人工提案；PPC 只复用去除图像的辅助函数，实际编辑器在 `ppc_proposer.py`。 |
| [library.py](../../rpent/improvement/library.py) | `ExperienceLibrary`、`LibrarySnapshot` | 旧协议的内容寻址经验库、证据校验和冻结检索；schema 2 主线在 `stream.py` 管理自己的任务后 lesson 前缀。 |
| [deployment.py](../../rpent/improvement/deployment.py) | `select_snapshot()`、`DeploymentSelection` | 依据旧资格报告、任务范围、预算和快照身份选择 candidate 或 fallback；不负责部署 Pi0.5 服务，也不选择 PPC final。 |

## 配置和测试入口

| 位置 | 覆盖范围 |
| --- | --- |
| [libero_memory.yaml](../../robots/libero/config/improvement/libero_memory.yaml) | 标准 LIBERO schema 2 模板：4 suite × 10 task，seed 0 构建，seed 1–10 评测；API 执行与编辑设置。 |
| [libero_pro_memory.yaml](../../robots/libero/config/improvement/libero_pro_memory.yaml) | LIBERO-Pro 的 task/swap 共 8 suite，沿用相同构建/评测划分。 |
| [windows_codex_remote.yaml](../../robots/libero/config/improvement/windows_codex_remote.yaml) | Windows 本地 Codex 与远端 worker 的 schema 2 模板；路径、参考语料来源和模型/checkpoint 身份需填实。它与已跑通 baseline 的独立启动脚本不是同一实验配置。 |
| [test_ppc.py](../../tests/unit_tests/rpent/improvement/test_ppc.py) | 四回合调度、独立候选、Compose、有限重试、原始证据投影、反思与无 lesson 情况。 |
| [test_stream.py](../../tests/unit_tests/rpent/improvement/test_stream.py) | 顺序构建、冻结前缀、跨 suite 检索、真实记录发布、冻结评测、初态重叠及配对拒绝。 |
| [test_snapshots.py](../../tests/unit_tests/rpent/memory/test_snapshots.py) | 叶子 schema、原子批量编辑、旧 hash、索引保留、语料篡改、读取审计与原生记录发布边界。 |
| [test_improvement.py](../../tests/unit_tests/robots/libero/test_improvement.py)、[test_env_selection.py](../../tests/unit_tests/robots/libero/test_env_selection.py) | 原生成功、预算、初态/reset、只读语料、记录导出和自有/借用服务清理。 |
| [test_remote_lifecycle.py](../../tests/unit_tests/robots/libero/test_remote_lifecycle.py)、[test_ssh.py](../../tests/unit_tests/rpent/improvement/test_ssh.py) | 远端单回合生命周期、租约/取消、重放、传输校验、Windows 路径转换、SSH 配置和隧道。 |
| [test_codex_contracts.py](../../tests/unit_tests/rpent/planner/test_codex_contracts.py) | Codex tools-only 设置、SDK 事件契约及普通调用兼容性。 |
| [test_baseline_smoke.py](../../tests/unit_tests/robots/libero/test_baseline_smoke.py)、[test_deployment_inputs.py](../../tests/unit_tests/robots/libero/test_deployment_inputs.py) | baseline 链路证据、任务成败分离、固定资源与原始 memory 校验；不能证明 PPC 的效果。 |
| [test_controller.py](../../tests/unit_tests/rpent/improvement/test_controller.py)、[test_proposer.py](../../tests/unit_tests/rpent/improvement/test_proposer.py)、[test_library.py](../../tests/unit_tests/rpent/improvement/test_library.py) | 旧 schema 1 的资格门槛、单候选提案、经验库与部署选择兼容。 |

审查算法时可重点追踪：完整叶子源是否真正提供给编辑器、同一 base 是否用于 INV/DOM/Compose、
meta 前缀是否只来自以前任务、四回合初态是否一致、success 是否来自环境、冻结评测是否隔离
构建阶段，以及失败成本和未知实际步数是否如实保留。上述测试主要使用离线替身，真实性能结论
仍须来自服务器的构建与配对评测。

## 现有边界

当前执行器和结构化编辑器支持 API 与 Codex SDK；旧配置默认 API，新增 Windows/远端配置使用 Codex。
Codex SDK/MCP 接入复用现有 RPent planner，不等同于复现论文历史的文件交互协议。
原论文未公开的数值参数使用显式仓库复现设置，不能声称逐参数或绝对分数严格复现。
Windows 可以执行离线检查，也可运行 Codex、memory 编辑和实验调度，通过 SSH 连接 Linux worker。
2026-10-30 已在上述实现基线上验证本地 Codex → SSH worker → Pi0.5/SAM3 → LIBERO 的普通
baseline 单任务冒烟，`libero_object/task0/seed0` 原生成功，未启用 improvement/PPC。
记录见 `logs/git-convergence-20261030/verification.json` 和其 `smoke-final/baseline-smoke.json`；
配置为 `gpt-6.1-sol/xhigh`，不能视为论文模型的严格复现或完整 benchmark。
复跑指令见[普通 Harness-VLA 冒烟指南](harness_vla_codex_libero_smoke.md)。
PPC 的真实 memory 构建、完整矩阵与提升结论仍待服务器实验。

构建和评测要求新输出目录，未实现自动恢复。完整语料来源、真实初态身份及 checkpoint 身份必须保存。
HF 历史语料的 scene-seed 与多快照问题见实验对齐文档；本实现不自动赋予其历史成绩。

## 2026-10-20 离线验证记录

以下记录来自迁移阶段，不包含新增远端执行的真实验证。验证覆盖多叶子原子性、四阶段 PPC、独立上下文、跨 suite 反思前缀、真实记录来源、
冻结评测、初态别名拒绝、空语料 schema 和部分未知成本。Windows、Python 3.12.13 下执行：

```powershell
.venv/Scripts/python.exe -m pytest tests/unit_tests/rpent/improvement tests/unit_tests/rpent/memory/test_snapshots.py tests/unit_tests/robots/libero/test_improvement.py tests/unit_tests/robots/libero/test_env_selection.py -q --basetemp=logs/pytest-improvement-release --junitxml=logs/improvement-final.xml --tb=short
.venv/Scripts/ruff.exe check . --preview
.venv/Scripts/ruff.exe format --check .
```

结果为 **181 passed、1 skipped**；跳过项需要 Windows 软链接权限，另有第三方 event-loop 弃用提示。
API TestModel 使用本地 asyncio socketpair，需要在当前沙箱外执行；没有请求外部模型。
Ruff lint 与仓库规定的非 preview 格式检查通过，共 338 个 Python 文件。

额外 CLI、机器人发现、序列化及工具兼容检查为 **84 passed、1 failed**：
失败发生在既有测试创建 symlink 处，Windows 返回 WinError 1314；保留测试原样。
完整单元测试收集仍被 YAM 的 `fcntl` 和缺少 `torch` 阻断（1394 项已收集、2 项收集错误）。
`pre-commit run --all-files` 使用工作区缓存仍无法下载 GitHub hook，故没有声明该命令通过。
对应记录为 `logs/improvement-compat-final.xml`、`logs/pytest-collection-current.txt` 和 `logs/precommit-current.txt`。

四份研究 Markdown 的本地链接有效，两份配置能离线解析为 400/800 个评测 case。
未生成 `docs/build`，未改动对外中英文文档。当时尚无真实 LIBERO 任务成功率或 memory 增益结果。
