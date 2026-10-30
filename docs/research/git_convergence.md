# 本地与服务器的 Git 版本收敛

代码以 `git@github.com:0Ayanami/RPent_with_improvement_harness.git` 为统一 `origin`，本地和服务器使用同一个 `codex/improvement-harness-v1` 分支。官方 `git@github.com:RLinf/RPent.git` 保留为 `upstream`。服务器的全部操作限定在 `/home2/wanquan/RPent`。

版本收敛要求三方一致：GitHub 分支、本地 HEAD、服务器 HEAD 指向同一个提交；两端的 Git tree 相同、工作树干净，并在该提交上重跑普通 Harness-VLA 冒烟。`source_identity()` 是运行时的附加检查，不能代替整个 Git tree 的核验。

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

最终验证记录保存到 `logs/git-convergence-20261030/verification.json`，包括实际提交 SHA、tree、两端 remote 与干净状态、运行时源摘要、依赖与模型身份、冒烟输出目录及原生结果。该记录在提交完成后生成，不再写入受控文档并追加提交，避免“报告写回后 HEAD 又改变”的循环。

验收应同时核对 `baseline-smoke.json.status`、`native_success`、`episode/trial.json` 的清理与原始 memory 读取审计。模型名称、推理强度、资源 revision 和预算必须来自本次配置；此前单回合记录不能代替本次提交上的测试，也不能据此推断全量 benchmark 成功率。
