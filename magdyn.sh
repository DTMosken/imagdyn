#!/usr/bin/env bash
# magdyn — interactive menu (pass args for direct commands)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi

ENVFILE="${ROOT}/.magdyn_env"
if [[ -f "$ENVFILE" ]] && command -v conda >/dev/null 2>&1; then
  PREFENV="$(tr -d '[:space:]' <"$ENVFILE" || true)"
  if [[ -n "${PREFENV}" ]]; then
    echo "Using preferred conda env: ${PREFENV}"
    exec conda run -n "${PREFENV}" --no-capture-output python -m magdyn "$@"
  fi
fi

set +e
"$PY" -m magdyn "$@"
ec=$?
set -e
if [[ $ec -ne 0 ]]; then
  echo
  echo "If this failed due to missing packages, activate an env then re-run:"
  echo "  conda activate <env_name>"
  echo "  ./magdyn.sh"
  echo "Or open the menu and choose Environment to set a preferred conda env."
fi
exit "$ec"
