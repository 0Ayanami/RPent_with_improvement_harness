# 本地与服务器的 Git 版本收敛

代码以 `git@github.com:0Ayanami/RPent_with_improvement_harness.git` 为统一 `origin`，本地和服务器使用同一个 `codex/improvement-harness-v1` 分支。官方 `git@github.com:RLinf/RPent.git` 保留为 `upstream`。服务器的全部操作限定在 `/home2/wanquan/RPent`。

版本收敛要求三方一致：GitHub 分支、本地 HEAD、服务器 HEAD 指向同一个提交；两端的 Git tree 相同、工作树干净，并在该提交上重跑普通 Harness-VLA 冒烟。`source_identity()` 是运行时的附加检查，不能代替整个 Git tree 的核验。

## 当前采用的代码基线

2026-10-30 首次同步采用原本地基线 `ef70215467cf3acc7f71e9fc89914d8e707d338d`，加上迁移及部署提交 `6f8dfc3`、RPC 文档修正提交 `6a22efe`。原服务器的官方版本是 `efc392adc98a30045d8fb83c4097917f28e20a94`，它在相同基线之上另有 14 个提交。

```text
ef70215（原本地基线）
├── 14 个官方提交 → efc392a（原服务器版本，保存在备份分支）
└── 6f8dfc3 → 6a22efe（迁移和部署版本，已完成真实 baseline 冒烟）
```

当前分支尚未合并这 14 个官方提交。两端“版本收敛”表示采用相同的迁移版本，不表示以原服务器较新的官方版本为基线。未合并的提交包括 `53c7cb7` 的 HF memory 来源及固定记录重放、`66d9020` 的 LIBERO 转动时末端位置保持、`a3340f4` 的 yaw 收敛修复和 `d3efb4f` 的 OpenAI-compatible 模型推理强度转发。提交未合并不等于其中每项行为都不存在，是否与迁移代码重叠需要逐项审查。

如需升级官方基线，应先在独立分支处理这 14 个提交与当前 memory、Planner、LIBERO 代码的接口差异，再重新验证；不直接将旧服务器工作树或 stash 覆盖到当前分支。

## 同步前备份

首次切换时，服务器原官方 `main` 比本地基线领先 14 个提交，工作树还含文件复制带来的 CRLF 和执行权限差异。因此保留原历史和改动，不能将该混合工作树直接当成统一代码提交。

备份位于服务器的 `logs/git-convergence-20261030/`，包含 `server-before.bundle`、`server-worktree-before.tar.gz`、`server-before.patch` 和同步前的状态、引用、remote 配置。原提交另由 `codex/server-before-convergence-20261030` 分支保存，未提交源码另保存在带说明的 Git stash 中。权重、资源、虚拟环境、缓存和实验结果均保留原位，不使用 `git clean` 或 `stash --all`。

本地的同步前差异及 Git 引用备份位于同名日志目录。后续不要直接应用旧服务器 stash 到统一版本；如需找回差异，应先在独立分支核对。

## 日常同步与核验

本地在 `D:\research\RPent`，服务器在 `/home2/wanquan/RPent`。两端分别执行以下命令，先确认当前分支和工作树：

```bash
git remote -v
git branch --show-current
git status --short
git rev-parse HEAD
git rev-parse 'HEAD^{tree}'
```

在本地完成并提交实际改动后推送当前分支。服务器工作树干净时，再快进拉取同一分支：

```bash
cd /home2/wanquan/RPent
git switch codex/improvement-harness-v1
git pull --ff-only
git status --short
git rev-parse HEAD
git rev-parse 'HEAD^{tree}'
```

两端的提交和 tree 必须逐字一致，`git status --short` 必须为空。若有服务器独立修改，先备份和审查，不能直接覆盖；若无法快进，先检查历史分叉，不使用强制推送或硬重置来掩盖差异。`.gitattributes` 固定 shell 脚本为 LF，Linux 的执行权限及符号链接由 Git checkout 恢复。

## 从受控文件启动和复跑

部署配置已经纳入 `robots/libero/deployment/`：工作区环境、服务管理、固定权重/资源/原始 memory 下载校验、服务器依赖清单和 Windows 运行脚本。实际大文件、缓存与结果仍被忽略；无需提交它们。

每次代码更新后，先确认没有活动回合，再按[冒烟指南](harness_vla_codex_libero_smoke.md)重启本项目的 Pi0.5、SAM3 和 worker，使进程加载本次 checkout 的代码。服务器模型服务使用受控脚本管理：

```bash
cd /home2/wanquan/RPent
source robots/libero/deployment/environment.sh
.venv/bin/python robots/libero/deployment/manage_services.py status
```

Windows 先打开 SSH 隧道，再在另一个终端运行受控脚本：

```powershell
Set-Location D:\research\RPent
./robots/libero/deployment/run-smoke.ps1
```

完整服务启动、代理、原始 memory 和资源恢复步骤见冒烟指南。本次只验证普通 Harness-VLA，不运行 improvement/PPC；固定 memory 的校验和传输不是经验优化。

## 验证记录

首次同步的最终验证记录保存到 `logs/git-convergence-20261030/verification.json`，对应 `6a22efeebf3a6fe2a96836572c936642f585031a`，包括当时提交 SHA、tree、两端 remote 与干净状态、运行时源摘要、依赖与模型身份、冒烟输出目录及原生结果。该记录在提交完成后生成，作为该次测试的固定证据；后续文档或代码更新不能改写它来声称新提交已经完成相同测试。

验收应同时核对 `baseline-smoke.json.status`、`native_success`、`episode/trial.json` 的清理与原始 memory 读取审计。模型名称、推理强度、资源 revision 和预算必须来自本次配置；此前单回合记录不能代替本次提交上的测试，也不能据此推断全量 benchmark 成功率。

## 缓存与过程文件

缓存目录的名称不能用于判断其内容是否可以删除。本部署中以下目录仍有实际用途：

| 路径 | 保留原因 |
| --- | --- |
| `.venv/`、`rpent.egg-info/` | 已安装环境及 editable 安装元数据 |
| `checkpoints/`、`.cache/openpi/` | Pi0.5、SAM3 权重及 OpenPI tokenizer，运行时直接读取 |
| `.cache/huggingface/`、`resources/` | 固定资产与语料缓存；服务器 LIBERO 资产包含指向 HF snapshot 的链接 |
| `memory/`、`logs/harness-vla-baseline/memory/` | 实际运行使用的原始 memory 或指向它的链接 |
| `.cache/pre-commit/` | 本地格式和提交检查的已安装 hook 环境 |
| GPU 编译缓存、`.tmp/` 中的活动服务文件 | 加速模型运行，或由仍在运行的服务持有 |
| 真实冒烟目录、报告/XML、服务日志和 PID、Git 备份 | 验证、排错、进程管理和找回同步前改动的依据 |

安装完成后，在确认没有安装进程、虚拟环境不从缓存加载源码或解释器的前提下，可以清理 UV/Pip 的安装缓存。未受版本控制的 pytest 临时 fixture、`.pytest_cache`、`.ruff_cache` 和可重新生成的 Python 字节码也可以清理；保留相应的测试报告。仓库历史已跟踪根目录的 `.coverage`，本次保留该文件。后续安装或首次运行可能需要重新下载依赖或生成缓存。

不要对整个工作区执行 `git clean -fdx`。清理应使用明确的目标清单，先解析并检查真实路径；服务器的目标及其解析结果必须均位于 `/home2/wanquan/RPent` 内，目录链接不能作为递归清理的入口。`.gitignore` 的 `.cache/` 规则覆盖项目内各层同名缓存目录。每次清理的目标、文件量和结果保存在 `logs/workspace-maintenance-20261030/`，该目录与实验数据一同保持在 Git 之外。

2026-10-30 的清理已完成：本地删除 108 个目标、45,061 个文件，逻辑大小 1,214,228,739 字节；服务器删除 7 个目标、72,653 个文件，逻辑大小 11,553,431,167 字节。主要是 UV/Pip 安装缓存、测试 fixture 和字节码；服务器另删除了两个已被 Git 同步取代的源码传输归档及校验文件。逻辑大小不等同于实际释放空间，缓存中的硬链接可能共享磁盘内容。目标清单分别见 `cleanup-local.json`、`cleanup-server.json`；清理后已核验本地 CLI、服务器依赖导入、模型服务与 worker 健康，模型、资产、备份和真实冒烟报告保持可用。后续操作生成新的缓存属于正常行为。
