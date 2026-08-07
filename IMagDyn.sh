#!/usr/bin/env bash
# IMagDyn - interactive menu (pass args for direct commands)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi

ENVFILE="${ROOT}/.imagdyn_env"
if [[ ! -f "$ENVFILE" && -f "${ROOT}/.magdyn_env" ]]; then
  ENVFILE="${ROOT}/.magdyn_env"
fi

CONDA_BIN="${CONDA_EXE:-}"
if [[ -z "${CONDA_BIN}" ]] || [[ ! -x "${CONDA_BIN}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
  elif [[ -x "${HOME}/anaconda3/bin/conda" ]]; then
    CONDA_BIN="${HOME}/anaconda3/bin/conda"
  elif [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
    CONDA_BIN="${HOME}/miniconda3/bin/conda"
  else
    CONDA_BIN=""
  fi
fi

if [[ -f "$ENVFILE" ]] && [[ -n "${CONDA_BIN}" ]]; then
  PREFENV="$(tr -d '[:space:]' <"$ENVFILE" || true)"
  if [[ -n "${PREFENV}" ]]; then
    echo "Using preferred conda env: ${PREFENV}"
    exec "${CONDA_BIN}" run -n "${PREFENV}" --no-capture-output python -m imagdyn "$@"
  fi
fi

set +e
"$PY" -m imagdyn "$@"
ec=$?
set -e
if [[ $ec -ne 0 ]]; then
  echo
  echo "If this failed due to missing packages, activate an env then re-run:"
  echo "  conda activate <env_name>"
  echo "  ./IMagDyn.sh"
  echo "Or open the menu and choose Environment to set a preferred conda env."
fi
exit "$ec"
