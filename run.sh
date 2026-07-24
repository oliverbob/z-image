#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python"

# Ensure Python bootstrap is enabled by default so missing deps
# (including diffusers) are auto-installed via run.py.
: "${RUN_PYTHON_BOOTSTRAP:=1}"
export RUN_PYTHON_BOOTSTRAP

# On a 24GB GPU the text-to-image and image-edit models together can legitimately
# use ~22GB once the edit feature loads its second model. The stock 20GB watchdog
# threshold would then restart the whole stack mid-request. Raise it so only a
# genuine near-OOM condition (>24GB) triggers a restart.
: "${ZIMAGE_GPU_WATCHDOG_THRESHOLD_MB:=24000}"
export ZIMAGE_GPU_WATCHDOG_THRESHOLD_MB

if [[ "${RUN_PYTHON_BOOTSTRAP}" == "1" ]]; then
	if [[ ! -x "${VENV_PYTHON}" ]]; then
		echo "Creating virtual environment at ${VENV_DIR} ..."
		python3 -m venv "${VENV_DIR}"
	fi

	if ! "${VENV_PYTHON}" -c "import fastapi, uvicorn, torch, multipart, diffusers" >/dev/null 2>&1; then
		echo "Installing Python dependencies in ${VENV_DIR} (pip install -e .) ..."
		"${VENV_PYTHON}" -m pip install --upgrade pip
		"${VENV_PYTHON}" -m pip install -e "${ROOT_DIR}"
	fi
fi

exec python3 "${ROOT_DIR}/run.py" "$@"