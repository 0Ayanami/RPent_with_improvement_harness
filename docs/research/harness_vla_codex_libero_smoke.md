# Codex + GPU 服务器运行 Harness-VLA 的 LIBERO 冒烟

本指南使用 Windows 上的 Codex 做高层规划，由 `wanquan@gpu6` 的 `/home2/wanquan/RPent` 执行 LIBERO、Pi0.5、SAM3 和动作原语。运行一个普通 Harness-VLA episode，读取固定 memory，不运行 improvement/PPC、候选 memory、probe/compose、任务后回顾或 meta-library。

两端之间的远程 worker 只负责传递工具调用和产物。Pi0.5 内部的 action chunk 循环全部留在 GPU 服务器。memory 的快照用于只读传输和完整性检查，不会执行 memory 优化。

## 1. 工作区和运行位置

| 内容 | 位置 |
| --- | --- |
| Windows 仓库 | `D:\research\RPent` |
| SSH 连接 | `wanquan@222.20.97.104:22` |
| 服务器唯一工作区 | `/home2/wanquan/RPent` |
| 高层工具 worker | 服务器 `127.0.0.1:8870` |
| Pi0.5 | 服务器 `127.0.0.1:8801` |
| SAM3 | 服务器 `127.0.0.1:8802` |
| Pi0.5 / Env GPU | 物理 GPU `2` |
| SAM3 GPU | 物理 GPU `5` |

服务器上的虚拟环境、下载、缓存、临时文件、权重和日志都必须位于项目目录内。不要修改系统 Python、系统 CUDA、全局 shell 配置或其他用户的进程。GPU 编号使用部署记录中已经核验的可用设备。

可用下面的命令进入服务器工作区，并将交互 shell 的历史文件也放在工作区内：

```powershell
ssh -t wanquan@222.20.97.104 "cd /home2/wanquan/RPent && export HISTFILE=/home2/wanquan/RPent/.cache/bash_history && exec bash --noprofile --norc"
```

本机终端执行后文标注为 PowerShell 的命令；服务器终端执行 Bash 命令。新增远程 worker 和本地代码必须一致，运行时会比较源码摘要；同步代码后应重启 worker。

## 2. 服务器环境和锁定资源

可执行的环境、下载、服务和 Windows 冒烟脚本都随 Git 保存在 `robots/libero/deployment/`；`logs/deploy-20261030/` 仅保存实际运行记录。每个服务器终端先执行下面两条命令，环境脚本会进入并检查唯一工作区，同时把缓存、临时文件、模型下载和生成配置限定在项目内：

```bash
cd /home2/wanquan/RPent
source robots/libero/deployment/environment.sh
mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$HF_HOME" "$LIBERO_CONFIG_PATH"
```

脚本保留原有用户主目录，不改写 `HOME` 或全局配置。后续启动、下载和检查都使用这个环境。模型、assets 和原始 memory 的固定身份及校验值也保存在 Git 的 `resources.json`、`memory-manifest.json` 中，不依赖被忽略的日志脚本或已经下载好的大文件。

本次固定使用以下权重；SHA256 指完整权重文件内容：

| 组件 | 来源与 revision | 服务器路径 |
| --- | --- | --- |
| Pi0.5 LIBERO SFT | `RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT`，`6222623f635769bfc73c9472e29fab9b7fd8e027` | `checkpoints/pi05/` |
| SAM3 | ModelScope `facebook/sam3`，`f8ee5d16d5c30d0e87b1d22cbd358ce97761df56` | `checkpoints/sam3/sam3.pt` |

预期文件摘要为：

```text
Pi0.5 model.safetensors
4d9089c941793f170b625c2ed0ac7a3aa09b6f103e52dbbc82e67301529d6683
SAM3 sam3.pt
9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
```

Pi0.5 还需要 `metadata.pt` 和 `physical-intelligence/libero/norm_stats.json`，不能只保留主权重文件。运行时需要的其他模型资源使用项目内缓存。环境参数设置 `PI05_CHECKPOINT_PATH` 为 checkpoint 目录，`SAM3_CHECKPOINT_PATH` 为 `sam3.pt` 文件。

本次标准 LIBERO 的 `.[libero]` 已安装完成，共 307 个包，不能与 PRO/PLUS 混装。原始环境及工具说明见 [LIBERO 指南](../source-zh/rst_source/simulators/libero.rst)。实际安装输出保存在 `logs/deploy-20261030/install-libero.log`，安装命令和临时网络设置保存在 `install-command.json`，解析后的依赖清单保存在 `installed-requirements.txt`。

| 组件 | 本次实际版本 |
| --- | --- |
| Python | `3.11.15` |
| Torch / torchvision | `2.7.1` / `0.22.1` |
| MuJoCo | `3.3.0` |
| RLinf | `0.4` |
| RPent OpenPI | `0.2.2` |

后续复用当前 `.venv`，不要重新创建或覆盖它。恢复 Linux Python 3.11 环境时，使用随 Git 保存的 `robots/libero/deployment/requirements-server.txt`；它固定了实际安装的包版本和外部 Git commit，并去掉指向旧工作区的 editable RPent 行。RPent 本身单独从当前 checkout 安装：

```bash
source robots/libero/deployment/environment.sh
if [ ! -x .venv/bin/python ]; then
  /home2/wanquan/.local/bin/uv venv --python 3.11 .venv
fi
/home2/wanquan/.local/bin/uv pip install --python .venv/bin/python --no-deps -r robots/libero/deployment/requirements-server.txt
/home2/wanquan/.local/bin/uv pip install --python .venv/bin/python --no-deps -e .
```

该清单用于本次 Linux GPU 环境，不在 Windows 安装。`--no-deps` 配合完整清单避免重新解析浮动分支；安装完成后用 `uv pip check --python .venv/bin/python` 核对依赖。

仅在重新安装依赖且服务器无法直接访问 GitHub 源码时，Windows 先单独打开这个下载代理隧道：

```powershell
ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=yes -R 127.0.0.1:18889:127.0.0.1:7890 wanquan@222.20.97.104
```

该命令把服务器回环端口 18889 转到本机已经运行的 HTTP 代理 7890。它用于获取依赖源码，服务器模型推理和后续冒烟不需要这个下载代理。服务器在前面 workspace 环境的基础上执行安装：

```bash
HTTP_PROXY=http://127.0.0.1:18889 \
HTTPS_PROXY=http://127.0.0.1:18889 \
NO_PROXY=127.0.0.1,localhost,mirrors.aliyun.com \
GIT_CONFIG_COUNT=2 \
GIT_CONFIG_KEY_1=http.https://github.com/.proxy \
GIT_CONFIG_VALUE_1=http://127.0.0.1:18889 \
/home2/wanquan/.local/bin/uv pip install \
  --python /home2/wanquan/RPent/.venv/bin/python \
  --no-deps -r robots/libero/deployment/requirements-server.txt --index-url https://mirrors.aliyun.com/pypi/simple
/home2/wanquan/.local/bin/uv pip install --python .venv/bin/python --no-deps -e .
```

阿里 PyPI 镜像直连，GitHub 源码请求走临时代理。Git 设置通过本条命令的环境变量传入，不修改用户全局 Git 配置；`environment.sh` 的索引 0 设置只跳过 OpenPI 示例中不使用的 submodule，正式 LIBERO 包仍由 extra 单独安装。已有可用环境不必重复安装。

权重下载和摘要校验由 Git 中的下载器执行；它读取固定清单，把权重及统计写入 `checkpoints/`，把 tokenizer 写入 `.cache/openpi/`。断点续传文件同样留在项目内：

```bash
.venv/bin/python -m robots.libero.deployment.fetch_artifacts --group models
cat logs/deploy-20261030/checkpoints.json
```

`checkpoints.json` 只有在全部下载及校验成功后才会生成，记录文件大小、SHA256 和具体下载 URL。本次四个权重及统计文件和一个 tokenizer 文件均已完成实际摘要校验。脚本已存在的文件仍会重新计算摘要。执行 `--group models --check-only` 可只校验已有文件；后续复用同一环境和模型时可直接跳过下载。

OpenPI 的数据缓存使用 `OPENPI_DATA_HOME`，不能只设置 `HF_HOME` 或 `XDG_CACHE_HOME`。环境脚本将它设为项目的 `.cache/openpi`，包含运行时下载的 tokenizer 等资源。

LIBERO 资源固定为数据集 `RLinf/LIBERO-assets` 的 commit `3ba78404b48e8c70fa4ac9782d5aac8e5b46d55f`，共 586 个文件。本次 `resources/libero` 及其解析后的目标都在 workspace 内，见 `assets-provenance.json`。当前已下载并接入环境，无需重新执行；只有在首次准备资源时使用：

```bash
.venv/bin/python -m robots.libero.deployment.fetch_artifacts --group assets
.venv/bin/libero-download-assets --link /home2/wanquan/RPent/resources/libero --skip-existing
```

`LIBERO_CONFIG_PATH` 和 `LIBERO_ASSET_PATH` 已由环境脚本指向项目内的配置和资源目录，不使用用户主目录下的默认 LIBERO 配置。

## 3. 启动和检查服务器服务

本次已经启动的服务可以直接复用，不要重复启动同一端口。服务器重启或需要重现时，分别打开三个服务器终端；每个终端均先进入工作区并加载环境脚本，再以前台方式启动对应服务。

三个终端都先设置与本次部署一致的 GPU 排序和 CPU 线程数：

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
```

Pi0.5：

```bash
cd /home2/wanquan/RPent
source robots/libero/deployment/environment.sh
CUDA_VISIBLE_DEVICES=2 \
PI05_CHECKPOINT_PATH=/home2/wanquan/RPent/checkpoints/pi05 \
.venv/bin/python -m rpent.robots.components.pi05_vla_server \
  --embodiment libero --cuda-device 2 \
  --transport http --host 127.0.0.1 --port 8801
```

SAM3：

```bash
cd /home2/wanquan/RPent
source robots/libero/deployment/environment.sh
CUDA_VISIBLE_DEVICES=5 \
SAM3_CHECKPOINT_PATH=/home2/wanquan/RPent/checkpoints/sam3/sam3.pt \
.venv/bin/python -m rpent.robots.components.sam3_server \
  --cuda-device 5 --transport http --host 127.0.0.1 --port 8802
```

等待两个模型加载后，启动高层工具 worker：

```bash
cd /home2/wanquan/RPent
source robots/libero/deployment/environment.sh
CUDA_VISIBLE_DEVICES=2 \
.venv/bin/python -m robots.libero.remote_worker \
  --host 127.0.0.1 --port 8870 \
  --work-dir logs/harness-vla-baseline/worker
```

worker 为每个 episode 启动相应 suite/task/seed 的 Env，规划在 Windows 执行。不要额外手动创建一个固定 task 的 Env 服务供所有任务共用。模型服务由操作者持有，worker 只清理自己启动的 Env。

在另一个服务器终端检查已加载服务；这些检查不替代后面的真实模型动作：

```bash
cd /home2/wanquan/RPent
source robots/libero/deployment/environment.sh
.venv/bin/python -c "from rpent.utils.rpc.http_rpc import HttpRpcClient; print(HttpRpcClient('http://127.0.0.1:8801').call('vla.status', timeout_s=10)); print(HttpRpcClient('http://127.0.0.1:8802').call('healthz', timeout_s=10))"
curl --fail --noproxy '*' http://127.0.0.1:8870/health
```

Pi0.5 本次 preset 返回 `action_chunk: 5`。worker 返回 `protocol: 1`、源码摘要和 `active_trial`；开始新回合前应为空闲状态。源码摘要不匹配时先同步两端最终代码并重启 worker，不要绕过检查。

前台服务的终端需要保持运行。停止时先结束本地 episode、等待远端清理，再在 worker、SAM3、Pi0.5 对应终端按 `Ctrl+C`。如果使用部署时的后台 PID 文件，应先核对 PID 对应的完整命令属于本项目，再停止该进程；不要使用笼统的 `pkill python`。

Git 中的后台模型服务脚本读取同一份环境配置，保留本次实际 PID 和日志目录，可在 workspace 中使用：

```bash
source robots/libero/deployment/environment.sh
.venv/bin/python -m robots.libero.deployment.manage_services status
# 仅在模型服务未运行时启动；程序检查现有 PID 和端口。
.venv/bin/python -m robots.libero.deployment.manage_services start
```

该脚本只管理 Pi0.5 和 SAM3，worker 仍按上面的命令启动。模型日志分别为 `logs/deploy-20261030/pi05.log`、`sam3.log`。需要释放模型 GPU 时，在 episode 和 worker 都已结束后执行 `manage_services.py stop`；脚本先核对 PID 的 Python 路径、模块和工作目录，再发送停止信号，不对其他进程发信号。本次后台 worker 的日志和 PID 文件为 `logs/harness-vla-baseline/worker.log`、`worker.pid`。

## 4. Windows 准备与 SSH 隧道

在 Windows 仓库根目录检查本地 Python 和 Codex 登录。服务器不需要 Codex 登录或 OpenAI API Key。新 checkout 尚无本地环境时，先安装基础包，不安装 Linux GPU 依赖清单：

```powershell
if (-not (Test-Path .venv/Scripts/python.exe)) { uv venv --python 3.12 .venv }
uv pip install --python .venv/Scripts/python.exe -e .
```

```powershell
Set-Location D:\research\RPent
Get-Command ssh, codex
codex login status
.venv/Scripts/python.exe -m robots.libero.baseline_smoke --help
```

如果 CLI 没有可用登录，执行 `codex login` 并完成浏览器认证。本次本地账号的 SDK 模型列表包含 `gpt-6.1-sol`，后面的实际执行命令使用该模型和 `xhigh`。模型名不使用 API Planner 的 `openai:` 前缀。使用订阅登录时，实际可用模型和额度以自己的账号为准。

本机 Windows 已启用 `127.0.0.1:7890` 系统代理，但新启动的 PowerShell 不会自动将它作为 SDK 的代理。因此，运行 Codex 的终端 B 先显式导出下面的环境变量。换电脑时使用那台机器真实可用的代理地址；无需代理的网络可以省略两条代理设置，仍保留回环绕过规则。

```powershell
$env:HTTP_PROXY = 'http://127.0.0.1:7890'
$env:HTTPS_PROXY = 'http://127.0.0.1:7890'
$env:NO_PROXY = '127.0.0.1,localhost'
.venv/Scripts/python.exe -m rpent.cli.check_llm --planner codex --model gpt-6.1-sol --timeout-s 60 --json
```

本次该检查实际返回 `ok: true`、`reply: "ok"`，耗时约 6.656 秒，使用 ChatGPT 登录，未配置 API Key。这一步会请求一次真实模型，仅证明认证、模型权限和云端网络可用。worker、MCP 和 SSH 隧道的回环通信必须绕过代理。

程序为本次明确接入的 RPent MCP 工具设置专属授权策略，允许已授权的原语调用连续运行，同时保留只读 sandbox 和原生命令限制。无需修改自己的全局 Codex 权限配置，也无需逐原语点击审批。该策略不扩展到其他 MCP 集成。

本次固定 memory 来自原公开 `GPT_5.5_xhigh` 语料，生成模型与本次执行模型不同。因此这次运行验证部署链路，不能作为 GPT-5.5 原文配置下的成绩。后续正式对齐应选择有权限的论文执行模型，并在所有对照组固定同一模型、推理强度和 memory 来源。

语料锁定为 `RLinf/RPent-memory` 的完整 commit `47f301ef33bf8d0e8c770342f16b3c39c6a0e8c1`，版本 `GPT_5.5_xhigh`。本地目录为 `logs/harness-vla-baseline/memory/baseline_gpt55`，它只作为固定输入读取。

当前本地目录已经准备好，无需重复下载。新 Windows checkout 可直接运行 Git 中的下载器：

```powershell
.venv/Scripts/python.exe -m robots.libero.deployment.fetch_artifacts --group memory
```

下载器只从锁定 commit 获取原公开语料，并逐个核对 Git 中 443 个 SHA256；不会运行探索或 memory 编辑。已有文件内容不符时会报错并保留文件，已有额外文件时也会拒绝把它冒充原始语料。`--group memory --check-only` 只校验现存内容。

若需要从公开来源重新准备，在服务器 workspace 环境中下载锁定的版本，再复制到 Windows。下面的输出目录只在尚不存在时创建：

```bash
.venv/bin/python -m robots.libero.deployment.fetch_artifacts --group memory --memory-dir memory/baseline_gpt55
```

Windows 侧的目标父目录应已存在，且 `baseline_gpt55` 尚不存在：

```powershell
New-Item -ItemType Directory -Force logs/harness-vla-baseline/memory
scp -P 22 -r wanquan@222.20.97.104:/home2/wanquan/RPent/memory/baseline_gpt55 ./logs/harness-vla-baseline/memory/
```

不要用 `--explore` 重新生成语料，也不要把我们的 improvement 构建产物作为这个基线的输入。

确认服务器三个服务启动后，在 Windows 终端 A 打开隧道并保持运行：

```powershell
ssh -N -T -p 22 -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=yes -L 127.0.0.1:8870:127.0.0.1:8870 wanquan@222.20.97.104
```

只转发 worker 的 8870 端口，不转发逐动作的 Env/VLA/SAM3 端口。第一次连接需核验并信任正确的 SSH 主机密钥；不要关闭主机密钥校验。终端 B 用于运行后面的本地冒烟命令。

## 5. 单回合冒烟入口和判定

入口为 `python -m robots.libero.baseline_smoke`，与 improvement 构建入口分开。它不调用编辑器或递归控制器。必须显式指定固定 memory，或使用 `--empty-memory` 标记空语料联通实验。空 memory 不能称为论文中的原始参考语料。

在已设置代理变量的 Windows 终端 B 执行下列单回合命令。服务器服务和终端 A 的 SSH 隧道保持运行。程序会自动创建独立的临时 Codex SDK 会话，无需手动新建桌面聊天；实验日志保存在输出目录。

当前采用普通运行的 100 轮、5000 秒和 10000 环境步上限。之前的 30 轮检查在任务恢复阶段耗尽预算，相关记录保留在 `smoke-003`，不能当作完整回合通过。

```powershell
$VlaIdentity = 'RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT@6222623f635769bfc73c9472e29fab9b7fd8e027; model.safetensors sha256:4d9089c941793f170b625c2ed0ac7a3aa09b6f103e52dbbc82e67301529d6683'
.venv/Scripts/python.exe -m robots.libero.baseline_smoke `
  --worker-endpoint http://127.0.0.1:8870 `
  --memory-dir logs/harness-vla-baseline/memory/baseline_gpt55 `
  --memory-provenance 'RLinf/RPent-memory@47f301ef33bf8d0e8c770342f16b3c39c6a0e8c1/libero/GPT_5.5_xhigh' `
  --suite libero_object --task 0 --seed 0 --libero-type standard `
  --model gpt-6.1-sol --reasoning-effort xhigh `
  --max-turns 100 --timeout-s 5000 --max-env-steps 10000 --max-vla-calls 240 `
  --vla-identity $VlaIdentity `
  --vla-endpoint http://127.0.0.1:8801 --sam3-endpoint http://127.0.0.1:8802 `
  --cuda-device 2 `
  --output-dir ("logs/harness-vla-baseline/smoke-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
```

模型 endpoint 是服务器视角的回环地址。`--cuda-device 2` 选择服务器的物理 GPU 2，不是 Windows 显卡。命令按时间创建新目录；已有记录均保留，不覆盖报告。

Git 中保存了相同参数的快捷脚本，自动设置本机代理并生成新输出目录，结束后恢复原进程环境变量。服务器服务和 SSH 隧道就绪后，也可执行：

```powershell
./robots/libero/deployment/run-smoke.ps1
```

命令退出码为 0 且 `baseline-smoke.json.status` 为 `passed`，表示以下链路证据齐全：

- 真实 Codex 会话通过普通 RPent 工具完成规划。
- 至少一个无错误的 `pi0_*` 命令执行了 VLA chunk，记录到 VLA 请求和环境动作。
- 有相机 PNG 观测、原生初态身份和环境结果。
- 规划器调用 `finish`，远端 toolkit 和自启 Env 清理完成。

`native_success` 单独记录任务是否完成。管线冒烟通过不等于任务成功率达标；规划器的成功文字或 `finish.status` 不能替代环境的原生成功信号。

查看结果：

```powershell
Get-Content <本次输出目录>/baseline-smoke.json
Get-Content <本次输出目录>/baseline-config.json
```

`admitted_env_steps` 按请求的完整动作块记账。底层环境在提前结束后可能补齐 action chunk，因此 `actual_env_steps` 可能为 `null`；这种情况不能把动作块长度当作已测得的真实物理步数。

每次必须使用新的 `--output-dir`。结果目录包含 `baseline-config.json`、`baseline-smoke.json`、`episode/trial.json`、`episode/transcript.json`、`episode/states.json`、图像和执行日志。原始 memory 的实际读取记录在 `episode/trial.json.memory_reads`，应单独检查，不能仅凭传输了语料就声称模型使用了它。服务器保留对应 session 的原始产物，断连或失败时应同时检查两端日志。

## 6. 已完成的验证与证据

2026-10-30 已完成权重、依赖和真实 GPU 组件检查。组件记录与完整源码的 Codex episode 记录分开，不能用一个组件的通过替代整条链路。

| 检查 | 实际结果 | 证据 |
| --- | --- | --- |
| Pi0.5 真实预测 | `[5, 7]`、`float32`、所有值有限 | `logs/deploy-20261030/component-check.json` |
| SAM3 真实分割 | `[256, 256]` mask，11,232 个像素，score `0.9921875` | 同上 |
| LIBERO reset 与渲染 | RGB `[256,256,3]`、深度 `[256,256]`、状态 8 维、相机内外参可读 | `env-component-check.json` |
| LIBERO 原生动作 | 实际执行一个 7 维动作并取得下一观测 | 同上 |
| Codex 云端连接 | 订阅登录，真实文本请求返回 `ok` | 本地 CLI 检查结果 |
| Codex → RPent MCP | 真实执行一次 `finish` handler，无错误，约 11.2 秒 | `logs/deploy-20261030/codex-mcp-policy-probe/result.json` |

独立组件检查不包含 memory 改进，也不代表 LIBERO 任务成功率。完整 episode 的 `baseline-smoke.json` 必须另外核验所有链路条件和 `native_success`。

## 7. 本次完整回合结果

2026-10-30 的 `logs/harness-vla-baseline/smoke-004` 已通过真实整链冒烟，LIBERO 原生任务也成功。任务为标准 `libero_object`、task 0、seed 0，语言为“pick up the alphabet soup and place it in the basket”。结果来自环境的 `libero.env.terminated`，不以规划器的文字判断代替。

| 项目 | 实测结果 |
| --- | --- |
| 冒烟状态 / 原生任务成功 | `passed` / `true` |
| 执行模型 / reasoning effort | `gpt-6.1-sol` / `xhigh` |
| 回合结束 / 清理 | `completed` / `cleanup_ok: true` |
| `finish` | `success` |
| Pi0.5 调用 / 记录的 VLA 原语 | 15 / 2 |
| 按动作块预算记账的环境步 | 835；真实物理步数未单独精确计量 |
| 环境 reset 次数 | 1 |
| 原始 memory 完整读取 | 9 次，均未截断；另有 3 次指南读取 |
| 高层请求计数 / 总耗时 | 30 / 约 510 秒 |
| memory 优化 | `adaptive_memory_enabled: false` |

证据文件：

- [整链报告](../../logs/harness-vla-baseline/smoke-004/baseline-smoke.json)。
- [实际配置与固定语料身份](../../logs/harness-vla-baseline/smoke-004/baseline-config.json)。
- [原生结果、memory 读取与清理审计](../../logs/harness-vla-baseline/smoke-004/episode/trial.json)。
- [动作与观测记录](../../logs/harness-vla-baseline/smoke-004/episode/states.json)。
- [Codex 输出](../../logs/harness-vla-baseline/smoke-004/episode/codex_object_t0_s0.txt)。

原始 memory 的完整内容摘要为 `sha256:5996d7db1b006cfd5f9a86fca8441f0dfe3a90bd1dd76084df28a37d4e9a8071`，回合后完整性检查通过，没有发布或改写经验。部分公开语料的任务编号或历史文件名与当前标准任务不匹配；Codex 在本次回合中放弃缺失的 task-specific 文件，读取现存的 global/task-family 经验，并以当前任务语言和相机观测为准。本次成功说明部署链路可用，不代表公开语料的任务编号已完成论文对齐。

调试过程的记录均保留：`smoke-001` 因 Windows 路径后缀转换错误取消，远端清理成功；`smoke-002` 因本地 SSH 隧道未恢复而未启动环境；`smoke-003` 正常读取原语料并执行 VLA，但在 30 轮上限处结束，没有 `finish`；`smoke-004` 使用修复后的传输和普通运行预算完成。前三次不能计为完整冒烟通过，也不用于估计成功率。

当前服务器的 Pi0.5、SAM3 与 worker 保持运行，可直接复用。复跑时先打开第 4 节的 SSH 隧道，再在另一个 Windows 终端执行 `./robots/libero/deployment/run-smoke.ps1`；它自动使用新目录。需要重启服务器服务时使用第 3 节命令。
