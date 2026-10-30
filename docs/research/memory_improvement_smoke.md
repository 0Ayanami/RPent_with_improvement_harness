# Memory improvement 单任务冒烟与开销记录

本次验证 Windows 本地 Codex 通过 SSH 调用 Linux GPU worker，完成一次真实的 memory 构建。范围是标准 LIBERO 的 `libero_object/task0/seed0`，从仅含 `# Memory` 的空语料开始，启用 k=1 PPC 与任务后反思。单任务没有跨任务 lesson 检索的机会；不用于估计成功率或证明 memory 带来提升。

## 配置和流程

执行模型与编辑模型均为 `gpt-6.1-sol/xhigh`，使用本地 Codex 登录。Pi0.5、SAM3 和仿真运行在 `/home2/wanquan/RPent`，端口分别为 8801、8802、8870。权重与原始参考语料沿用[普通冒烟指南](harness_vla_codex_libero_smoke.md)的固定 revision。

`suite_names: [libero_object]`、`task_ids: [0]` 限定一个构建任务；`evaluation_seeds: [1]` 只预声明冻结后的评测输入，本次不运行评测。正常构建包含 base、INV probe、DOM probe、composed final 四个独立回合、三阶段结构化编辑和一次反思。四回合使用同一原生初态。执行预算为每回合 100 个模型响应、5000 秒、10000 个准入环境步、240 次 VLA 调用；编辑器每次最多等待 600 秒。

产物中的 `construction_report.json`、`task_report.json` 和冻结 manifest 使用内容摘要封存。`ppc_task.json` 保存各回合、编辑尝试与反思结果，snapshot store 保存实际语料。运行结果与原生任务成功分别判断，不能用规划器文字或进程退出码代替环境证据。

## 计量口径

| 指标 | 来源与含义 |
| --- | --- |
| 执行 token | 四个 `TrialCost.input_tokens/output_tokens`；来自 SDK 已完成响应的累计用量 |
| 改进 token | INV、DOM、Compose 的所有尝试及反思用量，失败尝试也计入 |
| 全部 token | `total_cost.all_model_input_tokens/output_tokens`；执行与改进相加 |
| 请求数 | Codex SDK 观察到的模型响应数，不是工具调用数或外层会话数 |
| 回合延迟 | `TrialCost.wall_time_s`，包含启动、模型推理、工具、远程通信与产物回传 |
| 编辑延迟 | 每次 attempt 和反思的 `elapsed_s`，包括会话启动、推理与结果解析 |
| 总延迟 | `adaptation_wall_time_s` 为整个 PPC；`invocation.json.elapsed_s` 为命令端到端时间 |
| Codex 额度 | 实验前后 `account/rateLimits/read` 的账号百分比快照，仅在窗口 reset 相同时计算百分点变化 |

缓存输入是输入 token 的子集，推理输出是输出 token 的子集，不能再次相加。执行器 transcript 保留这两项子计数，编辑器当前没有持久化它们。美元成本未知，不以公开 API 单价折算 Codex 订阅费用。额度差可能包含本聊天及其他会话，且百分比有取整，不能作为本次实验的精确收费。

环境准入步数不是实际执行步数；后者不可测时保留 `null`。网络或请求中断产生的未知用量也保留未知及已知小计。

## 运行记录

2026-10-30 第一轮保存在 `logs/memory-improvement-smoke-20261030-01/`，代码为 `31413fe`。该轮未通过：本地远程 runner 未初始化输出目录，MCP 在替换工具 schema 的输出路径时失败，模型未执行机器人动作。已完成的两回合、两个编辑请求、清理、额度与失败成本均保留。

修复提交 `0cefbc4` 在远程 runner 拥有 trial 输出目录的边界调用现有 `init_output_dir()`。回归实际查询 MCP `tools/list`，检查输出路径可解析；19 项远程生命周期测试通过。另有 27 项构建流测试验证单 suite 选择及完整矩阵标记。第二轮保存到 `logs/memory-improvement-smoke-20261030-02/`，使用新目录重新开始，不覆盖或恢复失败构建。

## 实测结果

第二轮于北京时间 16:54:48 开始、17:26:04 结束，命令退出码为 0，15 项产物检查全部通过。代码为 `0cefbc4ec923d256affbf23c7f879d62371b1d95`，本地与服务器 tree 为 `7ffa7389fcff421dfae562f7eb0846202913eecd`。以下延迟按秒计，编辑阶段包含全部尝试。

| 阶段 | 延迟/秒 | 输入 token | 输出 token | 模型响应数 | VLA 调用 |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 302.61 | 1,090,794 | 5,629 | 18 | 8 |
| INV 编辑，两次尝试 | 185.88 | 24,716 | 4,456 | 2 | — |
| DOM 编辑，两次尝试 | 298.05 | 24,197 | 7,283 | 2 | — |
| INV probe | 270.12 | 1,129,767 | 5,240 | 20 | 9 |
| DOM probe | 306.91 | 1,306,313 | 5,737 | 22 | 8 |
| Compose，一次尝试 | 144.77 | 21,568 | 3,622 | 1 | — |
| final | 271.06 | 989,227 | 5,076 | 17 | 8 |
| 任务后反思 | 89.99 | 27,349 | 2,675 | 1 | — |

四个执行回合共 1,150.70 秒、4,516,101 输入 token、21,682 输出 token；编辑与反思共 718.67 秒、97,830 输入 token、18,036 输出 token。全部模型合计 **4,613,931 输入 token、39,718 输出 token、83 个模型响应**。PPC 总延迟 1,870.61 秒，命令端到端延迟 **1,875.65 秒，约 31 分 16 秒**。执行输入中有 4,192,896 个缓存 token，属于上述输入的子集；编辑器缓存细分未知。

四回合原生成功均为 `true`，来源均为 `libero.env.terminated`；初态身份相同，四次 reset，33 次 VLA 调用，四回合均执行 `finish` 并正常清理。实际环境步数未知，未以准入步数替代。

INV 新增 `global/terminal_before_motion_recovery.md`，DOM 新增 `task-family/soup_basket_held_insertion.md`。Compose 最终保留并修订后者；INV probe、DOM probe、final 的实际读取审计均包含各自新增的叶子。final 成功后，原生 audit/recipe 按现有发布接口写入 `task-specific/`。冻结语料共有 4 个文件：索引、一个 task-family 叶子、两个 task-specific 记录，内容摘要为 `sha256:75414b0ec5816b571a8369bd4dcb7dec3cfb137364b2f3e71c2bbce017e34f48`。

INV、DOM 的首次提案引用格式未通过语义校验，第二次成功，所有请求均计入开销。任务后反思有效，三条编辑观察均判为 `confounded`，返回 `lesson: null`，因此 meta-library 仍为空。单任务已验证反思结果持久化与无 lesson 分支，没有验证跨任务检索。四回合均成功也不能证明性能提高，尚未执行冻结后的 seed 1 对照评测。

第一轮失败尝试另消耗 145,871 输入 token、2,573 输出 token，命令耗时 193.09 秒。两次命令累计记录 **4,759,802 输入 token、42,291 输出 token、2,068.74 秒**，不包含排错、测试、同步、等待和本聊天的成本。

第二轮额度前快照为 16:54:23，后快照为 19:01:47，比命令结束晚约 96 分钟。5 小时窗口的 reset 已改变，差值保持 `null`；同一周窗口已用百分比从 10% 到 16%，增加 6 个百分点。这个变化覆盖账号其他活动和后续等待，不能归因为实验本身。第一轮额度快照及其差值也单独保留，没有用额度百分比反推 token 或费用。

精确配置、分阶段记录、汇总、额度窗口和复跑辅助脚本保存在两端同名的 `logs/memory-improvement-smoke-20261030-01/` 与 `02/`。本地保留完整 trial 产物；服务器保留 worker 的原始仿真产物，并同步独立的审计归档。重点文件为第二轮的 `summary.json`、`invocation.json`、`config.yaml`、`run/construction_report.json`、`run/frozen-ppc_meta.json` 和 `run/construction/libero_object_t0_s0/ppc/ppc_task.json`。

## 复跑

先按[普通冒烟指南](harness_vla_codex_libero_smoke.md)启动模型、worker 和 SSH 隧道，并核对源码身份。当前本地实验的精确配置与干净初始语料保存在上述第二轮目录。再次运行时复制配置到同一目录下的新文件，并将 `output_dir` 改为尚不存在的目录，例如 `run-repeat-01`；不覆盖既有运行或初始语料。

```powershell
Set-Location D:\research\RPent
$env:HTTP_PROXY = 'http://127.0.0.1:7890'
$env:HTTPS_PROXY = 'http://127.0.0.1:7890'
$env:NO_PROXY = '127.0.0.1,localhost'
.venv/Scripts/python.exe -m rpent.cli.improvement construct --config logs/memory-improvement-smoke-20261030-02/config-repeat.yaml
```

构建本身保存 token 和阶段耗时。`quota_snapshot.py` 可在运行前后分别写入不同的额度快照；该脚本使用安装的 SDK 低层只读接口，只保存额度窗口，不保存账号身份或认证信息。`summarize.py` 核验产物并汇总本次开销；对复跑目录使用 `--run-dir` 和新的 `--output`，同时确保命令耗时及额度快照对应同一轮运行。
