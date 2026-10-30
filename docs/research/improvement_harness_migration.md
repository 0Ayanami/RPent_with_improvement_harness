# Harness2 Memory 适配迁移记录

更新日期：2026-10-26。当前方法见[设计](improvement_harness_design.md)、
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

## 当前模块

| 模块 | 职责 |
| --- | --- |
| [ppc_models.py](../../rpent/improvement/ppc_models.py) | 多叶子 MemoryRevision、分阶段模型成本、逐编辑反思与可空 lesson |
| [ppc.py](../../rpent/improvement/ppc.py) | 四 rollout 的 k=1 调度、独立候选、有限重试、final 采用和证据引用校验 |
| [ppc_proposer.py](../../rpent/improvement/ppc_proposer.py) | API/Codex 独立结构化编辑、Compose 和反思，无工具权限 |
| [stream.py](../../rpent/improvement/stream.py) | seed 0 顺序构建、冻结前缀、真实任务记录发布、冻结评测与配对汇总 |
| [snapshots.py](../../rpent/memory/snapshots.py) | 原子多叶子事务、连续快照、原生 audit/recipe 发布与完整内容验证 |
| [LIBERO adapter](../../robots/libero/improvement.py) | 预算准入、真实初态/成功、记录导出和自有服务清理 |
| [远端接口](../../robots/libero/remote.py)、[worker](../../robots/libero/remote_worker.py) | 本地 Codex 调用远端原语、快照和产物校验传输、单 trial 和断连回收 |
| [SSH 配置入口](../../rpent/improvement/ssh.py) | setup、前台隧道、连接检查和有界真实 smoke |
| [CLI](../../rpent/cli/improvement.py) | plan、construct、evaluate、compare；安装入口 `rpent-improve` |
| [实验配置](../../robots/libero/config/improvement/libero_memory.yaml) | 标准 LIBERO 与 LIBERO-Pro 论文 task/seed 矩阵及显式模型/预算 |

原有 memory 叶子 schema、三层目录与读取工具继续使用。多叶子事务生成 v2 snapshot manifest；
旧 v1 和单叶子 API 兼容。模型不能编辑 task-specific，成功记录由环境终止与记录命令生成。
meta-library 与执行 corpus 分离，任务内只读以前任务的冻结前缀，评测不读取它。

## 与旧 MVP 的关系

此前的单叶子 patch → 原生成功改善门槛 → confirmation/regression 流程，
保留在 `controller.py`、`proposer.py`、`library.py` 和 `deployment.py`，仅供旧 schema 1 配置显式调用。
它的 accepted/rejected 报告不作为 PPC 的 final 选择器，旧测试结果也不作为新流程已经验证的证据。

新流程不使用 held-out seed 为提案或 Compose 选优；k=1 的 composed final 被采用不表示它已证明有效。
Native terminal 可以出现在普通工具观测中；私有 TrialResult 成功字段、资格报告和最终评测不进入改进上下文。

## 现有边界

当前执行器和结构化编辑器支持 API 与 Codex SDK；旧配置默认 API，新增 Windows/远端配置使用 Codex。
Codex SDK/MCP 接入复用现有 RPent planner，不等同于复现论文历史的文件交互协议。
原论文未公开的数值参数使用显式仓库复现设置，不能声称逐参数或绝对分数严格复现。
Windows 可以执行离线检查，也可运行 Codex、memory 编辑和实验调度，通过 SSH 连接 Linux worker。
SSH 登录已核验；GPU、环境资产、VLA checkpoint、worker 隧道和整矩阵运行仍需服务器验证。

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
未生成 `docs/build`，未改动对外中英文文档。尚无真实 LIBERO 任务成功率或 memory 增益结果。
