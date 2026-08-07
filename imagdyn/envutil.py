"""Dependency / environment helpers for the interactive CLI."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import paths

_ENV_FILE = paths.ROOT / ".magdyn_env"


def current_python() -> str:
    return sys.executable


def current_conda_env() -> str | None:
    return os.environ.get("CONDA_DEFAULT_ENV") or os.environ.get("VIRTUAL_ENV")


def load_preferred_env() -> str | None:
    try:
        name = _ENV_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return name or None


def save_preferred_env(name: str | None) -> None:
    if not name:
        try:
            _ENV_FILE.unlink(missing_ok=True)
        except OSError:
            pass
        return
    try:
        _ENV_FILE.write_text(name.strip() + "\n", encoding="utf-8")
    except OSError:
        pass


def list_conda_envs() -> list[str]:
    conda = shutil.which("conda")
    if not conda:
        return []
    try:
        out = subprocess.check_output(
            [conda, "env", "list", "--json"],
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    try:
        import json

        data = json.loads(out)
        envs = []
        for p in data.get("envs", []):
            envs.append(Path(p).name)
        return envs
    except Exception:
        return []


def format_missing_dep(exc: BaseException, *, lang: str = "zh") -> str:
    name = getattr(exc, "name", None) or ""
    msg = str(exc)
    if not name and "No module named" in msg:
        # No module named 'torch'
        parts = msg.split("'")
        if len(parts) >= 2:
            name = parts[1]
    cur = current_conda_env() or "(none)"
    py = current_python()
    if lang == "en":
        lines = [
            f"Missing dependency: {name or msg}",
            f"  Python: {py}",
            f"  Active env: {cur}",
            "  Fix: in your terminal, activate an env that has the packages, then re-run magdyn:",
            "    conda activate <env_name>",
            "    magdyn.cmd",
            "  Or use menu → Environment to set a preferred conda env (relaunches via conda run).",
        ]
    else:
        lines = [
            f"缺少依赖: {name or msg}",
            f"  当前 Python: {py}",
            f"  当前环境: {cur}",
            "  处理: 在终端切换到已安装依赖的环境后重新启动 magdyn：",
            "    conda activate <环境名>",
            "    magdyn.cmd",
            "  或在菜单「环境」中设置首选 conda 环境（将用 conda run 重新启动）。",
        ]
    return "\n".join(lines)


def relaunch_in_conda_env(env_name: str, argv: list[str] | None = None) -> int:
    """Re-exec this process via `conda run -n env python -m magdyn …`."""
    conda = shutil.which("conda")
    if not conda:
        print("conda not found on PATH.", file=sys.stderr)
        return 1
    args = list(argv) if argv is not None else list(sys.argv[1:])
    # Use env's `python` via conda run (not the current interpreter)
    cmd = [conda, "run", "-n", env_name, "--no-capture-output", "python", "-m", "magdyn", *args]
    print(f">>> {' '.join(cmd)}")
    try:
        return int(subprocess.call(cmd))
    except OSError as e:
        print(e, file=sys.stderr)
        return 1
