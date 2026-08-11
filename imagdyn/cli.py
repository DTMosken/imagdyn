"""Command-line interface: python -m imagdyn <cmd> …"""

from __future__ import annotations

import argparse
import http.server
import socketserver
import sys
import threading
import webbrowser
from typing import Callable


def _cmd_status(_: argparse.Namespace) -> int:
    from .assets import print_status, probe_assets, write_assets_json

    st = probe_assets()
    print_status(st)
    write_assets_json(st)
    return 0


def _cmd_ensure(args: argparse.Namespace) -> int:
    from .assets import ensure_derived_terrain, print_status

    st = ensure_derived_terrain(force=args.force, seed_template=not args.no_template)
    print_status(st)
    return 0


def _cmd_reshape(_: argparse.Namespace) -> int:
    try:
        from .reshape import main as reshape_main
    except ImportError:
        print(
            "reshape is local-only (imagdyn/reshape.py missing). "
            "It is gitignored and not part of the shared menu.",
            file=sys.stderr,
        )
        return 1
    reshape_main()
    return 0


def _cmd_contours(_: argparse.Namespace) -> int:
    try:
        from .contours import main as contours_main
    except (ImportError, ModuleNotFoundError) as e:
        from .envutil import format_missing_dep

        print(format_missing_dep(e, lang="zh"), file=sys.stderr)
        return 1
    contours_main()
    return 0


def _cmd_temperature(args: argparse.Namespace) -> int:
    argv_rest = list(args.temp_argv or [])
    old = sys.argv
    try:
        sys.argv = [old[0] + " temperature", *argv_rest]
        try:
            from .temperature import main as temperature_main
        except (ImportError, ModuleNotFoundError) as e:
            from .envutil import format_missing_dep

            print(format_missing_dep(e, lang="zh"), file=sys.stderr)
            return 1
        temperature_main()
    finally:
        sys.argv = old
    return 0


def _cmd_currents(args: argparse.Namespace) -> int:
    argv_rest = list(args.currents_argv or [])
    if argv_rest and argv_rest[0] == "--":
        argv_rest = argv_rest[1:]
    try:
        from .currents import main as currents_main
    except (ImportError, ModuleNotFoundError) as e:
        from .envutil import format_missing_dep

        print(format_missing_dep(e, lang="zh"), file=sys.stderr)
        return 1
    return int(currents_main(argv_rest))


def _cmd_wind(args: argparse.Namespace) -> int:
    argv_rest = list(args.wind_argv or [])
    if argv_rest and argv_rest[0] == "--":
        argv_rest = argv_rest[1:]
    try:
        from .wind import main as wind_main
    except (ImportError, ModuleNotFoundError) as e:
        from .envutil import format_missing_dep

        print(format_missing_dep(e, lang="zh"), file=sys.stderr)
        return 1
    return int(wind_main(argv_rest))


def _cmd_summarize(_: argparse.Namespace) -> int:
    try:
        from .summarize import main as summarize_main
    except (ImportError, ModuleNotFoundError) as e:
        from .envutil import format_missing_dep

        print(format_missing_dep(e, lang="zh"), file=sys.stderr)
        return 1
    summarize_main()
    return 0


def _cmd_viewer(args: argparse.Namespace) -> int:
    import atexit
    import signal

    from . import paths
    from .assets import ensure_derived_terrain, write_assets_json

    ensure_derived_terrain(seed_template=True)
    write_assets_json()

    port = args.port
    root = str(paths.ROOT)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=root, **kw)

        def log_message(self, fmt: str, *log_args) -> None:
            if args.verbose:
                super().log_message(fmt, *log_args)

    class ViewerServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    httpd = ViewerServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/viewer/"
    print(f"Serving {root}")
    print(f"Viewer: {url}")

    released = False
    release_lock = threading.Lock()

    def release_port() -> None:
        """Stop accept loop and close the listening socket (idempotent)."""
        nonlocal released
        with release_lock:
            if released:
                return
            released = True
        try:
            httpd.shutdown()
        except Exception:
            pass
        try:
            httpd.server_close()
        except Exception:
            pass

    atexit.register(release_port)

    def _on_signal(signum, frame) -> None:  # noqa: ARG001
        # shutdown() must not run on the serve_forever thread
        threading.Thread(target=release_port, daemon=True).start()

    prev_handlers: dict[int, object] = {}
    for sig in (
        signal.SIGINT,
        getattr(signal, "SIGTERM", None),
        getattr(signal, "SIGBREAK", None),
    ):
        if sig is None:
            continue
        try:
            prev_handlers[sig] = signal.signal(sig, _on_signal)
        except (ValueError, OSError):
            pass

    # Windows: console close / logoff — free the port before the process dies
    win_handler = None
    if sys.platform == "win32":
        import ctypes

        HandlerRoutine = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)

        def _console_ctrl(ctrl_type: int) -> int:
            # 0=C, 1=Break, 2=Close, 5=Logoff, 6=Shutdown
            if ctrl_type in (0, 1, 2, 5, 6):
                release_port()
                return 1
            return 0

        win_handler = HandlerRoutine(_console_ctrl)
        ctypes.windll.kernel32.SetConsoleCtrlHandler(win_handler, True)

    if not args.no_open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever(poll_interval=0.5)
    finally:
        release_port()
        for sig, handler in prev_handlers.items():
            try:
                signal.signal(sig, handler)  # type: ignore[arg-type]
            except (ValueError, OSError):
                pass
        if win_handler is not None:
            try:
                ctypes.windll.kernel32.SetConsoleCtrlHandler(win_handler, False)
            except Exception:
                pass
        try:
            atexit.unregister(release_port)
        except Exception:
            pass
        print("\nStopped.")
    return 0


def _cmd_pipeline(args: argparse.Namespace) -> int:
    """ensure → [reshape] → contours → [temperature] → summarize (if temps exist / requested)."""
    from .assets import ensure_derived_terrain, print_status, probe_assets, write_assets_json
    from .envutil import format_missing_dep
    from .timing import StepTimer, format_duration

    wall = StepTimer("pipeline")
    with wall.step("ensure"):
        ensure_derived_terrain(force=args.force, seed_template=not args.no_template)
    if args.reshape:
        try:
            from .reshape import main as reshape_main
        except ImportError:
            print(
                "pipeline --reshape skipped: imagdyn/reshape.py not present (local-only).",
                file=sys.stderr,
            )
        else:
            with wall.step("reshape"):
                reshape_main()
    try:
        from .contours import main as contours_main
    except (ImportError, ModuleNotFoundError) as e:
        print(format_missing_dep(e, lang="zh"), file=sys.stderr)
        return 1
    with wall.step("contours"):
        contours_main()
    if args.temperature:
        old = sys.argv
        try:
            sys.argv = [old[0]]
            try:
                from .temperature import main as temperature_main
            except (ImportError, ModuleNotFoundError) as e:
                print(format_missing_dep(e, lang="zh"), file=sys.stderr)
                return 1
            with wall.step("temperature"):
                temperature_main()
        finally:
            sys.argv = old
        try:
            from .summarize import main as summarize_main
        except (ImportError, ModuleNotFoundError) as e:
            print(format_missing_dep(e, lang="zh"), file=sys.stderr)
            return 1
        with wall.step("summarize"):
            summarize_main()
    else:
        write_assets_json()
        st = probe_assets()
        if st.temperature_annual:
            try:
                from .summarize import main as summarize_main
            except (ImportError, ModuleNotFoundError) as e:
                print(format_missing_dep(e, lang="zh"), file=sys.stderr)
                return 1
            with wall.step("summarize"):
                summarize_main()
        else:
            print("Skip summarize (no temperature maps). Pass --temperature to generate.")
    if getattr(args, "wind", False) and not args.temperature:
        try:
            from .wind import main as wind_main
        except (ImportError, ModuleNotFoundError) as e:
            print(format_missing_dep(e, lang="zh"), file=sys.stderr)
            return 1
        with wall.step("wind"):
            wind_main([])
    print_status()
    print(
        f"pipeline: {len(wall.steps)} stages  "
        f"avg/stage {format_duration(wall.mean_step)}  "
        f"total {format_duration(wall.elapsed)}"
    )
    print(wall.summary())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="imagdyn",
        description="IMagDyn — terrain / climate map tooling",
    )
    p.add_argument(
        "--lang",
        choices=("zh", "en"),
        default=None,
        help="Interactive menu language (zh/en); also used by `menu`",
    )
    sub = p.add_subparsers(dest="command", required=False)

    sp = sub.add_parser("menu", help="Interactive menu (中文 / English)")
    sp.add_argument("--lang", choices=("zh", "en"), default=None, help="Menu language")
    sp.set_defaults(func=_cmd_menu)

    sp = sub.add_parser("status", help="List which graph assets exist")
    sp.set_defaults(func=_cmd_status)

    sp = sub.add_parser(
        "ensure",
        help="If only Full Elevation exists, generate Land Mask + Above/Below (+ assets.json)",
    )
    sp.add_argument("--force", action="store_true", help="Regenerate derived terrain even if present")
    sp.add_argument("--no-template", action="store_true", help="Do not seed from graphs/template/")
    sp.set_defaults(func=_cmd_ensure)

    sp = sub.add_parser(
        "reshape",
        help="[local-only] Nonlinear remap + filters (requires imagdyn/reshape.py)",
    )
    sp.set_defaults(func=_cmd_reshape)

    sp = sub.add_parser("contours", help="Generate Terrain - Contours.png")
    sp.set_defaults(func=_cmd_contours)

    sp = sub.add_parser("temperature", help="Generate monthly / annual temperature maps")
    sp.add_argument(
        "temp_argv",
        nargs=argparse.REMAINDER,
        help="Args forwarded to temperature generator (e.g. --cpu --downsample 2)",
    )
    sp.set_defaults(func=_cmd_temperature)

    sp = sub.add_parser(
        "currents",
        help="Ocean current coastline filter (east=warm / west=cold; diagnostic maps)",
    )
    sp.add_argument(
        "currents_argv",
        nargs=argparse.REMAINDER,
        help="Args forwarded to imagdyn.currents (e.g. --dump-maps --cpu)",
    )
    sp.set_defaults(func=_cmd_currents)

    sp = sub.add_parser("wind", help="Generate wind / pressure fields from temperature")
    sp.add_argument(
        "wind_argv",
        nargs=argparse.REMAINDER,
        help="Args forwarded to imagdyn.wind (e.g. --cpu --annual-only)",
    )
    sp.set_defaults(func=_cmd_wind)

    sp = sub.add_parser(
        "summarize",
        help="Write temperature_stats.json and wind_stats.json (incl. waterworld 1D)",
    )
    sp.set_defaults(func=_cmd_summarize)

    sp = sub.add_parser("viewer", help="Serve project root and open the map viewer")
    sp.add_argument("--port", type=int, default=8765)
    sp.add_argument("--no-open", action="store_true", help="Do not open a browser")
    sp.add_argument("-v", "--verbose", action="store_true")
    sp.set_defaults(func=_cmd_viewer)

    sp = sub.add_parser("pipeline", help="ensure → contours → optional temperature/summarize")
    sp.add_argument("--force", action="store_true")
    sp.add_argument("--no-template", action="store_true")
    sp.add_argument("--reshape", action="store_true", help="Also run reshape before contours")
    sp.add_argument("--temperature", action="store_true", help="Also generate temperatures")
    sp.add_argument("--wind", action="store_true", help="Also generate wind/pressure (needs temperatures)")
    sp.set_defaults(func=_cmd_pipeline)

    return p


def _cmd_menu(args: argparse.Namespace) -> int:
    from .interactive import interactive_menu

    lang = getattr(args, "lang", None)
    return interactive_menu(lang=lang)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    # No args / only --lang → interactive menu (includes viewer)
    if not argv or (len(argv) == 2 and argv[0] == "--lang" and argv[1] in ("zh", "en")):
        from .interactive import interactive_menu

        lang = argv[1] if len(argv) == 2 else None
        return interactive_menu(lang=lang)
    if len(argv) == 1 and argv[0] in ("--lang=zh", "--lang=en"):
        from .interactive import interactive_menu

        return interactive_menu(lang=argv[0].split("=", 1)[1])

    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        from .interactive import interactive_menu

        return interactive_menu(lang=args.lang)
    if args.command == "temperature" and args.temp_argv and args.temp_argv[0] == "--":
        args.temp_argv = args.temp_argv[1:]
    if args.command == "currents" and args.currents_argv and args.currents_argv[0] == "--":
        args.currents_argv = args.currents_argv[1:]
    if args.command == "wind" and args.wind_argv and args.wind_argv[0] == "--":
        args.wind_argv = args.wind_argv[1:]
    func: Callable[[argparse.Namespace], int] = args.func
    return int(func(args))


if __name__ == "__main__":
    raise SystemExit(main())
