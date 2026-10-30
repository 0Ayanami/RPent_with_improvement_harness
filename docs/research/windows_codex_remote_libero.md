# Windows 本地 Codex 与 Linux GPU 服务器上的 LIBERO 实验

本指南在 Windows 上运行实验调度、Codex Planner、memory 编辑和结果统计，在 Linux GPU 服务器上运行 LIBERO、Pi0.5、SAM3 及原语执行。两端通过 SSH 隧道通信，服务器无需登录 Codex 或访问 OpenAI。

本地程序通过 Codex SDK 创建独立会话，复用本地 Codex 登录；它不会把批量实验交给当前桌面聊天窗口。账号可用模型和额度遵循实际订阅或提供方设置，参见 [Codex 认证说明](https://learn.chatgpt.com/docs/auth)。方法和实验边界分别见[研究设计](improvement_harness_design.md)与[实验对齐说明](libero_memory_experiment_alignment.md)。

2026-10-26 已通过只读 SSH 核验服务器：`/home2/wanquan/RPent` 存在且 Git 工作区干净，已有 Python 3.11.15 的 `.venv` 和基础 RPent/Codex SDK；服务器有 7 张 RTX 3090，每张 24 GiB。该环境尚未安装 Torch、MuJoCo、RLinf、OpenPI 和 SAM3，新增 worker 尚未同步，两个 checkpoint 环境变量也未设置，因此还不能直接运行 GPU 实验。本次未修改服务器。下文继续使用已有 `.venv`，补齐资源后再启动服务。

## 1. 部署位置和已知连接信息

```text
Windows：Codex SDK / Planner / 编辑器 / memory / 实验产物
    │ HTTP，通过本地 127.0.0.1:8870 进入 SSH 隧道
    ▼
Linux：remote_worker 127.0.0.1:8870
    ├── 每个 trial 独立的 LIBERO Env
    ├── 原语和 VLA chunk 内部循环
    ├── Pi0.5 127.0.0.1:8801
    └── SAM3 127.0.0.1:8802
```

每轮云端规划可以调用一个原语或 VLA 工具；该工具内部的连续动作在服务器执行。Windows 不需要安装 MuJoCo、LIBERO 或 GPU 模型依赖，也不需要转发 8801、8802。

| 项目 | 当前配置 |
| --- | --- |
| 本地仓库 | `D:\research\RPent` |
| SSH 目标 | `wanquan@222.20.97.104` |
| SSH 端口 | `22` |
| 服务器仓库 | `/home2/wanquan/RPent` |
| worker 本地/远端端口 | `8870` / `8870` |
| 本地配置目录 | `logs/local-libero/` |

SSH 地址和服务器目录已核验，本机默认 OpenSSH 认证可在严格主机密钥校验下登录，无需修改本地 `~/.ssh/config`。checkpoint 实际位置、模型可用性与 GPU 排期仍需准备。配置文件不保存密码或私钥内容。

## 2. Windows 准备和 Codex 登录

以下 PowerShell 命令均从仓库根目录执行。现有 `.venv` 可继续使用；安装基础包即可，不在 Windows 安装 `.[libero]`。

本机 `codex login status` 已确认 ChatGPT 登录，可先复用该登录。Codex 的用户缓存和运行文件仍可能写入系统盘；当前 C 盘空间不足，实验前需释放可用空间，即使项目位于 D 盘。

```powershell
Set-Location D:\research\RPent
uv pip install --python .venv/Scripts/python.exe -e .
Get-Command codex, ssh
codex login status
```

若 CLI 没有登录，执行 `codex login`，使用自己的账号完成浏览器认证。桌面软件与 SDK 的登录能否共享取决于当前用户、Codex 配置和凭据存储；以本地 CLI 和下一条检查的实际结果为准。不要把登录文件复制到服务器。

下面的检查会真实请求一次云端模型，只检查认证、模型权限和网络，不运行仿真：

```powershell
.venv/Scripts/python.exe -m rpent.cli.check_llm --planner codex --model gpt-5.5
```

模板固定使用 `gpt-5.5`、`xhigh`，与项目中的对应复现配置一致。使用其他模型时，需要同时修改执行器、评测执行器和编辑器，并在所有对照组保持一致。Codex 模型名直接写 `gpt-5.5`，不加 API Planner 的 `openai:` 前缀。

Codex 配置不填写 `base_url` 字段，自定义提供方通过现有 `CODEX_BASE_URL` 环境配置接入。当前 SDK 无请求前的硬输出 token 上限；`editor.max_tokens` 在响应返回后检查实际用量，超限会拒绝采用该响应，已经产生的云端用量仍计入成本。

使用现有账号登录不需要额外填写 `OPENAI_API_KEY`。若环境中已经配置 `CODEX_BASE_URL`、`CODEX_API_KEY`、`CODEX_BIN` 或自定义 Codex provider，应核对它们是否符合本次实验的实际提供方，避免误用旧代理配置。

## 3. 生成本地配置

本工作站的配置目录为 `logs/local-libero/`。已有 `connection.yaml` 和两个实验 YAML 时直接编辑，跳过下面的新建命令。首次配置或需要独立新实验时，选择一个尚不存在的目录：

```powershell
.venv/Scripts/python.exe -m rpent.improvement.ssh setup --directory logs/local-libero --host 222.20.97.104 --user wanquan --server-project /home2/wanquan/RPent
```

该命令生成：

| 文件或目录 | 用途 |
| --- | --- |
| `connection.yaml` | SSH 主机、端口、私钥路径和隧道端口 |
| `experiment.yaml` | 有跨任务 meta-library 的构建配置 |
| `experiment-no-meta.yaml` | 不读取跨任务 meta-library 的消融配置 |
| `memory/initial/MEMORY.md` | 空的初始 memory |
| `memory/reference/MEMORY.md` | 待替换的参考语料占位 |
| `runs/meta/`、`runs/no-meta/` | 对应实验运行时创建的产物目录 |

`setup` 拒绝覆盖已有目录。已有配置直接编辑；需要独立新实验时，换用新的目录名。所有实验路径相对各自 YAML 文件解析。

私钥登录可以在首次生成时追加 `--identity-file C:/Users/你的用户名/.ssh/rpent_gpu_ed25519`，或在 `connection.yaml` 的 `identity_file` 填写现有私钥绝对路径。服务器需预先授权对应公钥。密码登录则保留 `identity_file: null`，由 OpenSSH 在终端中询问密码。

## 4. 核验 SSH 和同步代码

先进行一次交互登录，核对服务器主机指纹，并确认工作目录：

```powershell
ssh -p 22 wanquan@222.20.97.104
```

进入服务器后执行：

```bash
hostname
cd /home2/wanquan/RPent
pwd
nvidia-smi
```

使用密钥时，首次登录命令也加 `-i` 和相同的私钥路径。后续隧道启用严格主机密钥校验，需要这次登录先将经核验的主机密钥写入本地 `known_hosts`。主机指纹发生变化时，先核实服务器变更原因。

两端必须使用相同的 RPent 代码。同步本次修改时应包含 `rpent/`、`robots/` 和 `pyproject.toml`；不要同步 Windows `.venv`、Codex 登录凭据或本地 GPU 权重。已有服务器仓库可以在本地另一终端执行：

```powershell
scp -P 22 -r ./rpent ./robots ./pyproject.toml wanquan@222.20.97.104:/home2/wanquan/RPent/
```

同步前应确认远端仓库没有需要保留的独立修改；该命令会更新同名文件，不会清理已删除的旧文件。也可使用自己的 Git 或文件同步工具更新完整 checkout。后文 `check` 会比较当前两端的源文件摘要；不一致时先完成代码同步，再重启 worker。

## 5. 服务器安装和资源

后续本节命令在 Linux 服务器执行，复用已存在的 Python 3.11 `.venv`，不重新创建或覆盖它。按 [LIBERO 安装说明](../source-zh/rst_source/simulators/libero.rst)准备 NVIDIA/CUDA、EGL、模型依赖和资源。服务器的 `uv` 在 `$HOME/.local/bin`，先补入当前 shell 的 PATH，再安装标准 LIBERO：

```bash
cd /home2/wanquan/RPent
export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate
uv pip install -e '.[libero]'
libero-download-assets --skip-existing
```

LIBERO-PRO 需使用独立环境安装 `.[libero-pro]` 并执行 `liberopro-download-assets --skip-existing`；对应配置改为 `benchmark: libero_pro`、`robot_options.libero_type: pro`。不同 LIBERO 版本不要混装在同一 Python 环境。

按上述 LIBERO 指南准备 `RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT` 的完整 checkpoint 和 `sam3.pt`。服务器若不能访问模型下载网站，可提前下载后同步。下面的路径必须换成服务器实际路径：

```bash
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PI05_CHECKPOINT_PATH=/实际路径/rlinf-pi05-libero-130-fullshot-sft
export SAM3_CHECKPOINT_PATH=/实际路径/sam3/sam3.pt
```

`PI05_CHECKPOINT_PATH` 指目录，`SAM3_CHECKPOINT_PATH` 指文件。这些变量只需要设置在服务器启动模型服务的会话中。GPU 显存需能同时容纳实际部署的 Pi0.5 与 SAM3；本地 Windows 的显卡不承担这些推理。示例选择 GPU 0，实际使用前按服务器资源安排选择可用 GPU，保持相关会话的设置一致。

## 6. 在服务器启动三个服务

建议在三个独立 `tmux` 会话中运行，保证 SSH 登录窗口退出后服务仍在。每个会话都先进入服务器仓库、激活相同 Python 环境，并设置需要的环境变量。下面命令使用服务器回环地址。

Pi0.5 会话：

```bash
tmux new -s rpent-vla
cd /home2/wanquan/RPent
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export PI05_CHECKPOINT_PATH=/实际路径/rlinf-pi05-libero-130-fullshot-sft
python -m rpent.robots.components.pi05_vla_server --embodiment libero --cuda-device 0 --transport http --host 127.0.0.1 --port 8801
```

按 `Ctrl+B` 后按 `D` 脱离会话，再启动 SAM3 会话：

```bash
tmux new -s rpent-sam3
cd /home2/wanquan/RPent
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export SAM3_CHECKPOINT_PATH=/实际路径/sam3/sam3.pt
python -m rpent.robots.components.sam3_server --cuda-device 0 --transport http --host 127.0.0.1 --port 8802
```

等待两个模型服务就绪后，脱离会话并启动 worker：

```bash
tmux new -s rpent-worker
cd /home2/wanquan/RPent
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python -m robots.libero.remote_worker --host 127.0.0.1 --port 8870 --work-dir logs/remote-worker
```

worker 为每个 trial 创建相应 suite/task/seed 的 Env，连接 8801/8802 模型服务。不要手动固定一个 Env 服务供全部任务共享。`execution.robot_options` 中两个模型 endpoint 都是服务器视角的地址。

查看服务使用 `tmux attach -t rpent-worker` 等命令。结束实验后，在对应会话中用 `Ctrl+C` 停止服务；worker 不拥有手动启动的模型服务。

## 7. Windows 打开 SSH 隧道和连接检查

Windows 终端 A 运行下列命令并保持窗口打开：

```powershell
.venv/Scripts/python.exe -m rpent.improvement.ssh tunnel --config logs/local-libero/connection.yaml
```

它调用本机 OpenSSH，建立 `127.0.0.1:8870 → 服务器127.0.0.1:8870` 转发，启用端口绑定失败检查和保活。可以先追加 `--dry-run` 查看完整命令。默认配置等价于：

```powershell
ssh -N -T -p 22 -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=yes -L 127.0.0.1:8870:127.0.0.1:8870 wanquan@222.20.97.104
```

Windows 终端 B 回到仓库根目录，检查隧道后的 worker：

```powershell
.venv/Scripts/python.exe -m rpent.improvement.ssh check --config logs/local-libero/connection.yaml
```

成功后保存 `connection-check.json`。这一步检查 worker 协议、两端源代码身份及 worker 是否空闲，不请求 Codex，也不证明 GPU 模型或仿真已经可用。若本地 8870 被占用，修改 `connection.yaml.local_port`，同时修改两个实验配置的 `execution.remote.endpoint` 和 `evaluation_execution.remote.endpoint`。

## 8. 准备参考 memory 和实验身份

生成的空 `memory/reference/MEMORY.md` 仅供静态计划检查，不代表 Harness-VLA 原始语料。正式构建前必须准备真实参考 memory，并填写来源和 checkpoint 身份。

现有 `robots.libero.memory sync` 使用 POSIX 文件锁，应在 Linux 下载，然后复制完整语料到 Windows。例如在服务器上执行：

```bash
cd /home2/wanquan/RPent
source .venv/bin/activate
python -m robots.libero.memory sync --memory-version GPT_5.5_xhigh --revision main --output-dir memory/reference-ready
```

命令打印语料目录；解析后的完整 commit 保存在服务器 `memory/libero/.versions/` 下对应版本指针 JSON 的 `commit` 字段中。正式归档时记录这个 commit，不能只记 `main`；已有明确版本时，将命令中的 `main` 换成该完整 commit。输出目录必须尚不存在。再从 Windows 复制：

```powershell
scp -P 22 -r wanquan@222.20.97.104:/home2/wanquan/RPent/memory/reference-ready ./logs/local-libero/memory/reference-ready
```

在 `experiment.yaml` 和 `experiment-no-meta.yaml` 中同步填写：

- `reference_memory.path: memory/reference-ready`，以及真实来源、完整 revision、生成模型配置和已知 scene-seed 历史。
- `execution.vla_identity` 和 `evaluation_execution.vla_identity`：checkpoint 仓库、固定 revision、实际文件 SHA256。模板中两处执行配置必须保持一致。
- `initial_memory` 保持干净，仅有空 `MEMORY.md`，除非本次明确研究继承语料上的改进。

身份字段是操作者声明，仍需核对服务器模型启动记录和真实文件摘要。模板占位标记会阻止正式构建。公开历史语料直接用于对照时，比较的是当前环境下两份语料的效果；若要单独归因于构建算法，还需从相同初始语料重新运行原探索方法并控制构建预算。

## 9. 静态计划和一次真实 smoke

先生成静态计划，不需要服务器、隧道或模型请求：

本地已用占位输入完成一次检查，结果保存在 `runs/meta/plan-template.json`，仅用于核对配置和任务矩阵。填写真实参考语料和 checkpoint 身份后，再运行下面命令生成正式 `plan.json`。

```powershell
.venv/Scripts/python.exe -m rpent.cli.improvement plan --config logs/local-libero/experiment.yaml
```

计划保存在 `logs/local-libero/runs/meta/plan.json`，核对 task/seed 表、模型、语料摘要和预计 rollout。默认标准 LIBERO 覆盖 40 个构建任务、160 次构建 rollout、每组 400 次测试。

服务器三个服务及 SSH 隧道就绪后，再执行一次真实联通检查：

```powershell
.venv/Scripts/python.exe -m rpent.improvement.ssh smoke --config logs/local-libero/experiment.yaml --output-dir logs/local-libero/smoke-001
```

smoke 固定为标准 LIBERO 的 `libero_spatial`、task 0、seed 0，最多 3 轮、300 秒、200 个环境步和 2 次 VLA 调用，会使用真实 Codex 及 GPU 服务。检查 `smoke.json` 中的原生状态身份、结束原因和清理结果；允许因刻意缩小预算而结束，任务失败本身不等于管线故障。该结果不计入论文分数或正式构建。PRO 环境需使用匹配版本的实际实验配置做额外验证，不能用此标准 LIBERO smoke 代替。

smoke 输出目录必须是新目录。通过后才开始完整构建，避免首次连接故障消耗完整实验预算。

## 10. 构建、冻结评测和对照

以下命令均在 Windows 终端 B 执行，终端 A 的隧道和服务器服务保持运行。构建每个任务依次执行基线 memory 回合（base）、INV probe、DOM probe、compose final，并记录任务后回顾。base 使用任务开始时的 memory，与冻结评测的 `reference_memory` 对照组不同。

```powershell
.venv/Scripts/python.exe -m rpent.cli.improvement construct --config logs/local-libero/experiment.yaml
```

成功完成后分别评测参考语料和新语料：

```powershell
.venv/Scripts/python.exe -m rpent.cli.improvement evaluate --frozen logs/local-libero/runs/meta/frozen-reference.json --output-dir logs/local-libero/eval-reference
.venv/Scripts/python.exe -m rpent.cli.improvement evaluate --frozen logs/local-libero/runs/meta/frozen-ppc_meta.json --output-dir logs/local-libero/eval-ppc-meta
.venv/Scripts/python.exe -m rpent.cli.improvement compare --reports logs/local-libero/eval-reference/evaluation_report.json logs/local-libero/eval-ppc-meta/evaluation_report.json --output logs/local-libero/comparison.json
```

若进行无 meta-library 消融，保持两个配置的语料、模型、任务和预算相同，单独运行：

```powershell
.venv/Scripts/python.exe -m rpent.cli.improvement construct --config logs/local-libero/experiment-no-meta.yaml
.venv/Scripts/python.exe -m rpent.cli.improvement evaluate --frozen logs/local-libero/runs/no-meta/frozen-ppc_no_meta.json --output-dir logs/local-libero/eval-ppc-no-meta
.venv/Scripts/python.exe -m rpent.cli.improvement compare --reports logs/local-libero/eval-reference/evaluation_report.json logs/local-libero/eval-ppc-meta/evaluation_report.json logs/local-libero/eval-ppc-no-meta/evaluation_report.json --output logs/local-libero/comparison-with-ablation.json
```

评测不调用编辑器，不更新 memory。任务成功来自环境原生判定；Codex 调用 `finish` 或自然语言宣布成功都不能替代它。不同 seed 的真实初态必须互不重复，并与构建 seed 0 分离；管线会核对这些身份，检查失败时不得删掉相关 case 缩小分母。

## 11. 产物、断连和运行限制

- Windows `runs/meta/`：计划、不可变 memory 快照、构建轨迹、回顾库、成本和冻结文件。
- Windows `eval-*/`：逐回合证据和 `evaluation_report.json`；对照结果在 `comparison*.json`。
- Linux `logs/remote-worker/`：远端 trial 及原始执行产物。保留它们，尤其是断连和基础设施故障发生时。

worker 同时处理一个 trial。长实验保持 Windows 不休眠、Codex 网络可用、SSH 隧道存活；`Ctrl+C` 停止隧道不会停止手动启动的服务器模型进程。断连后先查看 worker 和本地产物，等待当前 trial 完成清理或超时回收，再重新检查。第一版不支持从已有 construction 目录恢复，失败后保留原目录，使用新的实验输出目录重跑。

完整构建会产生多次 Codex 调用和远端 rollout，应预留账号额度及本地/远端磁盘空间。原语调用耗时不能超过 `remote.request_timeout_s`；实际任务需要更长单次工具执行时，统一增加对照组该配置，并相应调整 episode 预算。

远程 memory 快照使用兼容 Windows 的相对文件路径，最多传输 10,000 个文件、64 MiB 内容。若使用自建语料，应先检查文件名和大小；冻结文件及其内部路径、hash 不应手工改写后冒充原实验。

本地离线检查和模拟 worker 测试只说明接口与隔离行为。SSH 登录已核验；GPU 模型加载、LIBERO 执行和 memory 性能提升仍需在补齐资源的服务器上逐步验证，以 smoke 和正式报告保存的证据为准。

## 12. 本次验证记录

2026-10-26 已完成本地配置生成、Codex ChatGPT 登录状态检查和真实 SSH 只读连接。已确认服务器目录 `/home2/wanquan/RPent` 与现有 Python 环境，尚未修改服务器或加载 GPU 模型。

组合离线测试覆盖 improvement、memory snapshot、LIBERO trial/remote、环境选择、Codex 和 CLI 合约：310 通过、1 跳过、2 失败。两项失败均为既有 Windows 用例：创建符号链接缺少系统权限，以及测试要求同时保留大小写不同的 `NO_PROXY`/`no_proxy`，但 Windows 将它们视为同一个环境变量。相关测试未被删除或放宽。运行命令为：

```powershell
$env:NO_PROXY = '127.0.0.1,localhost'
.venv/Scripts/python.exe -m pytest tests/unit_tests/rpent/improvement tests/unit_tests/rpent/memory/test_snapshots.py tests/unit_tests/robots/libero/test_improvement.py tests/unit_tests/robots/libero/test_remote_lifecycle.py tests/unit_tests/robots/libero/test_env_selection.py tests/unit_tests/rpent/planner/test_codex_contracts.py tests/unit_tests/rpent/cli/test_main_contracts.py -q
```

Windows asyncio 与模拟 HTTP 服务需要本地回环连接权限；测试中的 Codex、仿真和模型均使用离线替身。全仓库 Ruff 检查与格式检查通过（343 个 Python 文件），`git diff --check` 通过。上述验证不包含真实 Codex 模型调用、GPU smoke 或 benchmark 成功率。
