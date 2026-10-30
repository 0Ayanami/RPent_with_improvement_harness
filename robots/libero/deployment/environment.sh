#!/usr/bin/env bash
# Source this file before Linux deployment commands. Keep every mutable path in RPent.
RPENT_DEPLOYMENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)" || { return 1 2>/dev/null || exit 1; }
if [[ "$RPENT_DEPLOYMENT_ROOT" != /home2/wanquan/RPent ]]; then
    echo "This deployment profile requires /home2/wanquan/RPent." >&2
    return 1 2>/dev/null || exit 1
fi
export RPENT_DEPLOYMENT_ROOT
cd -- "$RPENT_DEPLOYMENT_ROOT" || { return 1 2>/dev/null || exit 1; }
export UV_CACHE_DIR="${RPENT_DEPLOYMENT_ROOT}/.cache/uv"
export UV_PYTHON_INSTALL_DIR="${RPENT_DEPLOYMENT_ROOT}/.cache/uv-python"
export PIP_CACHE_DIR="${RPENT_DEPLOYMENT_ROOT}/.cache/pip"
export HF_HOME="${RPENT_DEPLOYMENT_ROOT}/.cache/huggingface"
export HF_HUB_CACHE="${RPENT_DEPLOYMENT_ROOT}/.cache/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="${RPENT_DEPLOYMENT_ROOT}/.cache/huggingface/hub"
export HF_DATASETS_CACHE="${RPENT_DEPLOYMENT_ROOT}/.cache/huggingface/datasets"
export XDG_CACHE_HOME="${RPENT_DEPLOYMENT_ROOT}/.cache/xdg"
export XDG_CONFIG_HOME="${RPENT_DEPLOYMENT_ROOT}/.config"
export XDG_DATA_HOME="${RPENT_DEPLOYMENT_ROOT}/.local/share"
export XDG_STATE_HOME="${RPENT_DEPLOYMENT_ROOT}/.local/state"
export TORCH_HOME="${RPENT_DEPLOYMENT_ROOT}/.cache/torch"
export TORCH_EXTENSIONS_DIR="${RPENT_DEPLOYMENT_ROOT}/.cache/torch_extensions"
export TRITON_CACHE_DIR="${RPENT_DEPLOYMENT_ROOT}/.cache/triton"
export CUDA_CACHE_PATH="${RPENT_DEPLOYMENT_ROOT}/.cache/cuda"
export MPLCONFIGDIR="${RPENT_DEPLOYMENT_ROOT}/.cache/matplotlib"
export NUMBA_CACHE_DIR="${RPENT_DEPLOYMENT_ROOT}/.cache/numba"
export PYTHONPYCACHEPREFIX="${RPENT_DEPLOYMENT_ROOT}/.cache/pycache"
export TMPDIR="${RPENT_DEPLOYMENT_ROOT}/.tmp"
export TMP="${RPENT_DEPLOYMENT_ROOT}/.tmp"
export TEMP="${RPENT_DEPLOYMENT_ROOT}/.tmp"
export RAY_TMPDIR="${RPENT_DEPLOYMENT_ROOT}/.tmp/ray"
export LIBERO_CONFIG_PATH="${RPENT_DEPLOYMENT_ROOT}/.config/libero"
export UV_LINK_MODE="copy"
export HF_HUB_DISABLE_XET="1"
export HF_HUB_DOWNLOAD_TIMEOUT="60"
export HF_HUB_ETAG_TIMEOUT="30"
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export ROBOT_PLATFORM="LIBERO"
export LIBERO_TYPE="standard"
export GIT_CONFIG_COUNT="1"
export GIT_CONFIG_KEY_0="submodule.active"
export GIT_CONFIG_VALUE_0="__rpent_no_optional_submodules__"
export GIT_LFS_SKIP_SMUDGE="1"
export OPENPI_DATA_HOME="${RPENT_DEPLOYMENT_ROOT}/.cache/openpi"
export RPENT_RLINF_ROOT="${RPENT_DEPLOYMENT_ROOT}/.venv/lib/python3.11/site-packages"
export __GL_SHADER_DISK_CACHE="0"
export __GL_SHADER_DISK_CACHE_PATH="${RPENT_DEPLOYMENT_ROOT}/.cache/nvidia/GLCache"
export IMAGEIO_USERDIR="${RPENT_DEPLOYMENT_ROOT}/.cache/imageio"
export JAX_COMPILATION_CACHE_DIR="${RPENT_DEPLOYMENT_ROOT}/.cache/jax"
export RPENT_LIBERO_VIDEO_DIR="${RPENT_DEPLOYMENT_ROOT}/.tmp/primitive_videos"
export LIBERO_ASSET_PATH="${RPENT_DEPLOYMENT_ROOT}/resources/libero"
export PI05_CHECKPOINT_PATH="${RPENT_DEPLOYMENT_ROOT}/checkpoints/pi05"
export SAM3_CHECKPOINT_PATH="${RPENT_DEPLOYMENT_ROOT}/checkpoints/sam3/sam3.pt"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
mkdir -p -- "$RPENT_DEPLOYMENT_ROOT/.cache" "$RPENT_DEPLOYMENT_ROOT/.config/libero" \
    "$RPENT_DEPLOYMENT_ROOT/.tmp/primitive_videos" "$RPENT_DEPLOYMENT_ROOT/logs/deploy-20261030"
