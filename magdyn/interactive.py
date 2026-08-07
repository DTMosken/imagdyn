"""Interactive bilingual menu for magdyn."""

from __future__ import annotations

import argparse
import os
import sys
import traceback

from . import paths
from .envutil import (
    current_conda_env,
    current_python,
    format_missing_dep,
    list_conda_envs,
    load_preferred_env,
    relaunch_in_conda_env,
    save_preferred_env,
)

_LANG_FILE = paths.ROOT / ".magdyn_lang"

TEXTS = {
    "zh": {
        "title": "magdyn 交互菜单",
        "pick_lang": "选择语言 / Choose language",
        "lang_zh": "中文",
        "lang_en": "English",
        "prompt": "请选择",
        "invalid": "无效选项，请重试。",
        "back": "按 Enter 返回菜单…",
        "bye": "再见。",
        "run": "执行",
        "done": "[完成]",
        "fail": "[退出码 {ec}]",
        "error": "出错（菜单不会退出）:",
        "items": [
            ("1", "status", "查看资源状态"),
            ("2", "ensure", "补全地形派生图 / 遮罩"),
            ("3", "contours", "生成等高线"),
            ("4", "temperature", "生成气温图"),
            ("5", "summarize", "统计气温 / 地形"),
            ("6", "viewer", "启动地图查看器"),
            ("7", "pipeline", "一键流水线"),
            ("8", "env", "环境（查看 / 切换 conda）"),
            ("9", "lang", "切换语言 (中/英)"),
            ("0", "exit", "退出"),
        ],
        "force_ensure": "强制重建派生地形? [y/N]: ",
        "force_cpu": "强制使用 CPU? [y/N]: ",
        "port": "端口 (默认 8765): ",
        "viewer_hint": "提示: Ctrl+C 停止服务器",
        "gen_temp": "生成气温图? [y/N]: ",
        "env_header": "当前运行环境",
        "env_python": "Python",
        "env_active": "活动环境",
        "env_preferred": "首选 conda 环境",
        "env_list": "可用 conda 环境",
        "env_prompt": "输入要切换的 conda 环境名（留空取消；- 清除首选）: ",
        "env_none": "(无)",
        "env_relaunch": "将用该环境重新启动 magdyn…",
        "env_no_conda": "未找到 conda，请在终端手动: conda activate <环境名>",
        "yn_yes": ("y", "Y", "是", "是的"),
    },
    "en": {
        "title": "magdyn interactive menu",
        "pick_lang": "Choose language / 选择语言",
        "lang_zh": "中文",
        "lang_en": "English",
        "prompt": "Select",
        "invalid": "Invalid option, try again.",
        "back": "Press Enter to return to menu…",
        "bye": "Bye.",
        "run": "Running",
        "done": "[done]",
        "fail": "[exit {ec}]",
        "error": "Error (menu stays open):",
        "items": [
            ("1", "status", "View asset status"),
            ("2", "ensure", "Derive mask / above / below"),
            ("3", "contours", "Generate contour map"),
            ("4", "temperature", "Generate temperature maps"),
            ("5", "summarize", "Climate / terrain stats"),
            ("6", "viewer", "Start map viewer"),
            ("7", "pipeline", "One-shot pipeline"),
            ("8", "env", "Environment (view / switch conda)"),
            ("9", "lang", "Switch language (ZH/EN)"),
            ("0", "exit", "Exit"),
        ],
        "force_ensure": "Force regenerate derived terrain? [y/N]: ",
        "force_cpu": "Force CPU? [y/N]: ",
        "port": "Port (default 8765): ",
        "viewer_hint": "Tip: Ctrl+C stops the server",
        "gen_temp": "Generate temperature maps? [y/N]: ",
        "env_header": "Current runtime",
        "env_python": "Python",
        "env_active": "Active env",
        "env_preferred": "Preferred conda env",
        "env_list": "Available conda envs",
        "env_prompt": "Conda env to switch to (empty=cancel; -=clear preferred): ",
        "env_none": "(none)",
        "env_relaunch": "Relaunching magdyn in that environment…",
        "env_no_conda": "conda not found. In a terminal: conda activate <env_name>",
        "yn_yes": ("y", "Y", "yes", "YES"),
    },
}


def _clear() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def _load_lang() -> str | None:
    try:
        lang = _LANG_FILE.read_text(encoding="utf-8").strip().lower()
    except OSError:
        return None
    return lang if lang in TEXTS else None


def _save_lang(lang: str) -> None:
    try:
        _LANG_FILE.write_text(lang + "\n", encoding="utf-8")
    except OSError:
        pass


def _ask_lang(default: str | None = None) -> str:
    print()
    print(f"  {TEXTS['zh']['pick_lang']}")
    print(f"  1) {TEXTS['zh']['lang_zh']}")
    print(f"  2) {TEXTS['en']['lang_en']}")
    print()
    hint = "1/2"
    if default == "zh":
        hint = "1/2, default 1"
    elif default == "en":
        hint = "1/2, default 2"
    raw = input(f"  [{hint}]: ").strip()
    if not raw and default in TEXTS:
        return default
    if raw in ("1", "zh", "ZH", "中文", "cn", "CN"):
        return "zh"
    if raw in ("2", "en", "EN", "english", "English"):
        return "en"
    return default if default in TEXTS else "zh"


def _yes(ans: str, lang: str) -> bool:
    a = (ans or "").strip()
    if not a:
        return False
    return a in TEXTS[lang]["yn_yes"] or a.lower() in ("y", "yes")


def _run_argv(argv: list[str], lang: str) -> int:
    from . import cli

    t = TEXTS[lang]
    print()
    print(f">>> {t['run']}: python -m magdyn {' '.join(argv)}")
    print()
    try:
        ec = int(cli.main(argv))
    except (ImportError, ModuleNotFoundError) as e:
        print(t["error"])
        print(format_missing_dep(e, lang=lang))
        return 1
    except SystemExit as e:
        code = e.code
        ec = int(code) if isinstance(code, int) else (1 if code else 0)
        if ec != 0:
            print(t["fail"].format(ec=ec))
        else:
            print(t["done"])
        return ec
    except Exception as e:
        print(t["error"])
        print(f"  {type(e).__name__}: {e}")
        # Keep menu alive; short traceback for debugging
        traceback.print_exc(limit=4)
        if isinstance(e, (ImportError, ModuleNotFoundError)) or "No module named" in str(e):
            print(format_missing_dep(e, lang=lang))
        else:
            cur = current_conda_env() or TEXTS[lang]["env_none"]
            if lang == "zh":
                print(f"  当前环境: {cur}")
                print("  可在终端 conda activate <环境> 后重新运行 magdyn.cmd")
            else:
                print(f"  Active env: {cur}")
                print("  Activate another env in your terminal, then re-run magdyn.")
        return 1
    print()
    print(t["done"] if ec == 0 else t["fail"].format(ec=ec))
    return ec


def _pause(lang: str) -> None:
    try:
        input(f"\n{TEXTS[lang]['back']}")
    except EOFError:
        pass


def _env_menu(lang: str) -> str | None:
    """Show env info; optionally set preferred conda env and relaunch. Returns 'relaunch' sentinel."""
    t = TEXTS[lang]
    print()
    print(f"=== {t['env_header']} ===")
    print(f"  {t['env_python']}: {current_python()}")
    print(f"  {t['env_active']}: {current_conda_env() or t['env_none']}")
    pref = load_preferred_env()
    print(f"  {t['env_preferred']}: {pref or t['env_none']}")
    envs = list_conda_envs()
    if envs:
        print(f"  {t['env_list']}: {', '.join(envs)}")
    else:
        print(f"  {t['env_no_conda']}")
    print()
    name = input(t["env_prompt"]).strip()
    if not name:
        return None
    if name == "-":
        save_preferred_env(None)
        return None
    save_preferred_env(name)
    print(t["env_relaunch"])
    # Relaunch interactive menu in that env
    argv = ["--lang", lang]
    ec = relaunch_in_conda_env(name, argv)
    # If relaunch returns, stay in this process only on failure
    if ec != 0:
        print(t["fail"].format(ec=ec))
        return None
    return "relaunch"


def interactive_menu(*, lang: str | None = None) -> int:
    """Run bilingual interactive menu. Returns process exit code."""
    if lang not in TEXTS:
        saved = _load_lang()
        lang = _ask_lang(saved or "zh")
        _save_lang(lang)

    while True:
        t = TEXTS[lang]
        _clear()
        print("=" * 40)
        print(f"  {t['title']}")
        print("=" * 40)
        pref = load_preferred_env()
        active = current_conda_env()
        if pref or active:
            print(f"  [{active or t['env_none']}]" + (f"  preferred={pref}" if pref else ""))
        print()
        for key, _cmd, label in t["items"]:
            print(f"  {key}) {label}")
        print()
        try:
            choice = input(f"{t['prompt']} [0-9]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print(t["bye"])
            return 0

        if choice in ("0", "q", "Q"):
            print(t["bye"])
            return 0

        if choice == "9":
            lang = "en" if lang == "zh" else "zh"
            _save_lang(lang)
            continue

        try:
            if choice == "1":
                _run_argv(["status"], lang)
            elif choice == "2":
                ans = input(t["force_ensure"])
                argv = ["ensure", "--force"] if _yes(ans, lang) else ["ensure"]
                _run_argv(argv, lang)
            elif choice == "3":
                _run_argv(["contours"], lang)
            elif choice == "4":
                extra: list[str] = []
                if _yes(input(t["force_cpu"]), lang):
                    extra.append("--cpu")
                argv = ["temperature", "--", *extra] if extra else ["temperature"]
                _run_argv(argv, lang)
            elif choice == "5":
                _run_argv(["summarize"], lang)
            elif choice == "6":
                port = input(t["port"]).strip() or "8765"
                print()
                print(t["viewer_hint"])
                _run_argv(["viewer", "--port", port], lang)
            elif choice == "7":
                argv = ["pipeline"]
                if _yes(input(t["force_ensure"]), lang):
                    argv.append("--force")
                if _yes(input(t["gen_temp"]), lang):
                    argv.append("--temperature")
                _run_argv(argv, lang)
            elif choice == "8":
                if _env_menu(lang) == "relaunch":
                    return 0
            else:
                print(t["invalid"])
                _pause(lang)
                continue
        except KeyboardInterrupt:
            print()
            # Don't exit menu on Ctrl+C during a job — return to menu
            print(t["error"] if lang == "zh" else t["error"])
            if lang == "zh":
                print("  已中断当前任务，返回菜单。")
            else:
                print("  Interrupted; back to menu.")
            _pause(lang)
            continue

        _pause(lang)


def main_menu_cmd(args: argparse.Namespace) -> int:
    return interactive_menu(lang=getattr(args, "lang", None))
