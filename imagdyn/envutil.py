"""Dependency / environment helpers for the interactive CLI."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import paths

_ENV_FILE = paths.ROOT / ".imagdyn_env"


def current_python() -> str:
    return sys.executable


def current_conda_env() -> str | None:
    return os.environ.get("CONDA_DEFAULT_ENV") or os.environ.get("VIRTUAL_ENV")


def load_preferred_env() -> str | None:
    for path in (_ENV_FILE, paths.ROOT / ".magdyn_env"):
        try:
            name = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if name:
            return name
    return None


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


def find_conda() -> str | None:
    """
    Resolve a real conda executable.

    On Windows, ``conda`` is often a PowerShell function (Invoke-Conda), so
    ``shutil.which('conda')`` fails even when Anaconda is installed. Prefer
    ``CONDA_EXE`` and well-known install layouts.
    """
    candidates: list[Path] = []

    for key in ("CONDA_EXE", "CONDA_BAT"):
        raw = os.environ.get(key)
        if raw:
            candidates.append(Path(raw))

    which = shutil.which("conda")
    if which:
        candidates.append(Path(which))
    # Windows: conda.bat is often more reliable than a shim-less name
    which_bat = shutil.which("conda.bat")
    if which_bat:
        candidates.append(Path(which_bat))
    which_exe = shutil.which("conda.exe")
    if which_exe:
        candidates.append(Path(which_exe))

    prefixes: list[Path] = []
    for key in ("CONDA_ROOT", "CONDA_PREFIX", "CONDA_PREFIX_1"):
        raw = os.environ.get(key)
        if raw:
            prefixes.append(Path(raw))
    # If PREFIX is an env (…/envs/tf-gpu), also try its parent root
    for p in list(prefixes):
        if p.name and (p.parent / "Scripts" / "conda.exe").is_file():
            prefixes.append(p.parent)
        if p.parent.name == "envs":
            prefixes.append(p.parent.parent)

    home = Path.home()
    prefixes.extend(
        [
            Path(r"D:\ProgramData\anaconda3"),
            Path(r"C:\ProgramData\anaconda3"),
            Path(r"C:\ProgramData\miniconda3"),
            home / "anaconda3",
            home / "miniconda3",
            home / "mambaforge",
            home / "miniforge3",
            home / "AppData" / "Local" / "anaconda3",
            home / "AppData" / "Local" / "miniconda3",
            Path("/opt/anaconda3"),
            Path("/opt/miniconda3"),
            Path("/usr/local/anaconda3"),
            Path("/usr/local/miniconda3"),
        ]
    )

    for root in prefixes:
        candidates.append(root / "Scripts" / "conda.exe")
        candidates.append(root / "Scripts" / "conda.bat")
        candidates.append(root / "condabin" / "conda.bat")
        candidates.append(root / "condabin" / "conda")
        candidates.append(root / "bin" / "conda")

    seen: set[str] = set()
    for c in candidates:
        try:
            resolved = str(c.resolve()) if c.exists() else ""
        except OSError:
            continue
        if not resolved or resolved in seen:
            continue
        seen.add(resolved)
        if c.is_file():
            return str(c)
    return None


def list_conda_envs() -> list[str]:
    conda = find_conda()
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
            "  Fix: in your terminal, activate an env that has the packages, then re-run IMagDyn:",
            "    conda activate <env_name>",
            "    IMagDyn.cmd",
            "  Or use menu → Environment to set a preferred conda env (relaunches via conda run).",
        ]
    else:
        lines = [
            f"缺少依赖: {name or msg}",
            f"  当前 Python: {py}",
            f"  当前环境: {cur}",
            "  处理: 在终端切换到已安装依赖的环境后重新启动 IMagDyn：",
            "    conda activate <环境名>",
            "    IMagDyn.cmd",
            "  或在菜单「环境」中设置首选 conda 环境（将用 conda run 重新启动）。",
        ]
    return "\n".join(lines)


def relaunch_in_conda_env(env_name: str, argv: list[str] | None = None) -> int:
    """Re-exec this process via `conda run -n env python -m imagdyn …`."""
    conda = find_conda()
    if not conda:
        print(
            "conda not found.\n"
            "  Tried PATH, CONDA_EXE, and common Anaconda/Miniconda install paths.\n"
            "  Fix: open Anaconda Prompt, or set CONDA_EXE to …\\Scripts\\conda.exe,\n"
            "  then re-run IMagDyn / choose Environment again.",
            file=sys.stderr,
        )
        return 1
    args = list(argv) if argv is not None else list(sys.argv[1:])
    # Use env's `python` via conda run (not the current interpreter)
    cmd = [conda, "run", "-n", env_name, "--no-capture-output", "python", "-m", "imagdyn", *args]
    print(f">>> {' '.join(cmd)}")
    try:
        return int(subprocess.call(cmd))
    except OSError as e:
        print(e, file=sys.stderr)
        return 1
