import argparse
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class CommandSpec:
    raw: str
    argv: list[str]
    exp_id: str
    log_path: Path


@dataclass
class RunningProcess:
    spec: CommandSpec
    process: subprocess.Popen[str]
    log_f: object


ACTIVE_PROCESSES: list[RunningProcess] = []
ACTIVE_LOCK = threading.Lock()
SHUTDOWN_REQUESTED = False


def _register_process(handle: RunningProcess) -> None:
    with ACTIVE_LOCK:
        ACTIVE_PROCESSES.append(handle)


def _unregister_process(process: subprocess.Popen[str]) -> None:
    with ACTIVE_LOCK:
        ACTIVE_PROCESSES[:] = [h for h in ACTIVE_PROCESSES if h.process is not process]


def _snapshot_processes() -> list[RunningProcess]:
    with ACTIVE_LOCK:
        return list(ACTIVE_PROCESSES)


def _log_shutdown(log_f: object, reason: str) -> None:
    try:
        log_f.write(f"\nSHUTDOWN_REASON: {reason}\n")
        log_f.write(f"SHUTDOWN_EPOCH: {time.time()}\n")
        log_f.flush()
    except Exception:
        pass


def _wait_for_exit(handles: list[RunningProcess], timeout_s: float) -> list[RunningProcess]:
    deadline = time.time() + timeout_s
    remaining = handles
    while time.time() < deadline:
        remaining = [h for h in handles if h.process.poll() is None]
        if not remaining:
            return []
        time.sleep(0.2)
    return remaining


def _terminate_active(reason: str) -> None:
    handles = [h for h in _snapshot_processes() if h.process.poll() is None]
    if not handles:
        return
    print(f"[SHUTDOWN] {reason}: terminating {len(handles)} running process(es).")
    for handle in handles:
        _log_shutdown(handle.log_f, reason)
        try:
            handle.process.send_signal(signal.SIGINT)
        except Exception:
            pass
    remaining = _wait_for_exit(handles, 5.0)
    for handle in remaining:
        try:
            handle.process.terminate()
        except Exception:
            pass
    remaining = _wait_for_exit(remaining, 5.0)
    for handle in remaining:
        try:
            handle.process.kill()
        except Exception:
            pass


def _request_shutdown(signum: int, _frame) -> None:
    global SHUTDOWN_REQUESTED
    if SHUTDOWN_REQUESTED:
        return
    SHUTDOWN_REQUESTED = True
    _terminate_active(reason=f"SIGNAL_{signum}")


def _read_commands(path: Path) -> list[str]:
    commands: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        commands.append(stripped)
    return commands


def _get_flag_value(argv: list[str], flag: str) -> Optional[str]:
    if flag in argv:
        idx = argv.index(flag)
        if idx + 1 >= len(argv):
            return None
        return argv[idx + 1]
    prefix = f"{flag}="
    for arg in argv:
        if arg.startswith(prefix):
            return arg.split("=", 1)[1]
    return None


def _replace_python_executable(argv: list[str]) -> list[str]:
    if not argv:
        return argv
    if argv[0] in {"python", "python3"}:
        return [sys.executable, *argv[1:]]
    return argv


def _infer_script_path(argv: list[str], repo_root: Path) -> Optional[Path]:
    for arg in argv:
        if not arg.endswith(".py"):
            continue
        candidate = (repo_root / arg).resolve()
        if candidate.exists():
            return candidate
    return None


def _infer_algorithm(script_path: Optional[Path], argv: list[str]) -> str:
    if script_path is None:
        return "exp"
    name = script_path.stem.lower()
    if "baseline" in name:
        model = _get_flag_value(argv, "--model") or _get_flag_value(argv, "--algorithm")
        return model.lower() if model else "baseline"
    for algo in ("iql", "ippo", "qmix"):
        if algo in name:
            return algo
    return name


def _build_exp_id(
    argv: list[str],
    repo_root: Path,
) -> str:
    script_path = _infer_script_path(argv, repo_root)
    conditional = bool(script_path and script_path.name.startswith("cond_"))
    algorithm = _infer_algorithm(script_path, argv)

    network = _get_flag_value(argv, "--net") or "unk"
    alg_conf = _get_flag_value(argv, "--alg-conf") or "config1"
    env_conf = _get_flag_value(argv, "--env-conf") or "config1"
    task_conf = _get_flag_value(argv, "--task-conf") or "config1"
    env_seed = int(_get_flag_value(argv, "--env-seed") or 42)
    torch_seed_raw = _get_flag_value(argv, "--torch-seed")
    torch_seed = int(torch_seed_raw) if torch_seed_raw is not None else None

    try:
        sys.path.insert(0, str(repo_root))
        from scripts.utils import generate_exp_id  # type: ignore

        return generate_exp_id(
            algorithm=algorithm,
            network=network,
            alg_config=alg_conf,
            env_config=env_conf,
            task_config=task_conf,
            env_seed=env_seed,
            torch_seed=torch_seed,
            conditional=conditional,
            results_root=None,
        )
    except Exception:
        parts = [
            algorithm,
            network,
            f"a{alg_conf}",
            f"e{env_conf}",
            f"t{task_conf}",
            str(env_seed),
        ]
        if torch_seed is not None:
            parts.append(str(torch_seed))
        exp_id = "_".join(parts)
        if conditional:
            exp_id = f"c_{exp_id}"
        return exp_id


def _ensure_unique_id(exp_id: str, results_root: Path) -> str:
    if not (results_root / exp_id).exists():
        return exp_id
    suffix = 2
    while True:
        candidate = f"{exp_id}_run{suffix}"
        if not (results_root / candidate).exists():
            return candidate
        suffix += 1


def _ensure_id_and_log(
    raw: str,
    repo_root: Path,
    results_root: Path,
    log_name: str,
    unique_id: bool,
) -> CommandSpec:
    argv = shlex.split(raw)
    argv = _replace_python_executable(argv)

    existing_id = _get_flag_value(argv, "--id")
    exp_id = existing_id or _build_exp_id(argv, repo_root=repo_root)
    if (existing_id is None) and unique_id:
        exp_id = _ensure_unique_id(exp_id, results_root)
    if existing_id is None:
        argv = [*argv, "--id", exp_id]

    exp_dir = results_root / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_path = exp_dir / log_name
    return CommandSpec(raw=raw, argv=argv, exp_id=exp_id, log_path=log_path)


def _run_batch(batch: list[CommandSpec], repo_root: Path, env: dict[str, str]) -> list[int]:
    processes: list[tuple[CommandSpec, subprocess.Popen[str], object]] = []
    try:
        for spec in batch:
            if SHUTDOWN_REQUESTED:
                break
            log_f = spec.log_path.open("w", encoding="utf-8")
            cmd_str = shlex.join(spec.argv)
            log_f.write(f"COMMAND: {cmd_str}\n")
            log_f.write(f"START_EPOCH: {time.time()}\n\n")
            log_f.flush()

            process = subprocess.Popen(
                spec.argv,
                cwd=str(repo_root),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            _register_process(RunningProcess(spec=spec, process=process, log_f=log_f))
            processes.append((spec, process, log_f))
            print(f"[START] {spec.exp_id} -> {spec.log_path}")

        codes: list[int] = []
        for spec, process, log_f in processes:
            code = int(process.wait())
            codes.append(code)
            _unregister_process(process)
            log_f.write(f"\nEND_EPOCH: {time.time()}\n")
            log_f.write(f"RETURN_CODE: {code}\n")
            log_f.close()
            status = "OK" if code == 0 else f"FAIL({code})"
            print(f"[DONE]  {spec.exp_id} {status}")
        return codes
    finally:
        for _, process, log_f in processes:
            if getattr(process, "poll", lambda: None)() is None:
                try:
                    process.kill()
                except Exception:
                    pass
            _unregister_process(process)
            try:
                log_f.close()
            except Exception:
                pass


def main() -> int:
    signal.signal(signal.SIGINT, _request_shutdown)
    signal.signal(signal.SIGTERM, _request_shutdown)

    parser = argparse.ArgumentParser(
        description="Run the commands in a todo file in parallel batches (default: 4 at a time)."
    )
    parser.add_argument("todo", nargs="?", default="todo.txt", help="Path to todo file (default: todo.txt).")
    parser.add_argument("--jobs", type=int, default=4, help="Number of commands to run in parallel per batch.")
    parser.add_argument(
        "--results-root",
        type=str,
        default="results",
        help="Results directory used by the OpenURB scripts (default: results).",
    )
    parser.add_argument(
        "--log-name",
        type=str,
        default="stdout.log",
        help="Log filename created inside results/<exp_id>/ (default: stdout.log).",
    )
    parser.add_argument(
        "--unique-id",
        action="store_true",
        help="Append _runN when an exp_id already exists (prevents overwriting).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print resolved commands and exit.")
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop after a batch if any command in it fails.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    todo_path = (repo_root / args.todo).resolve() if not os.path.isabs(args.todo) else Path(args.todo).resolve()
    results_root = (repo_root / args.results_root).resolve()

    commands = _read_commands(todo_path)
    if not commands:
        print(f"No commands found in {todo_path}")
        return 0

    specs = [
        _ensure_id_and_log(
            raw,
            repo_root=repo_root,
            results_root=results_root,
            log_name=args.log_name,
            unique_id=args.unique_id,
        )
        for raw in commands
    ]

    if args.dry_run:
        for spec in specs:
            print(f"{shlex.join(spec.argv)}\n  log: {spec.log_path}")
        return 0

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")

    jobs = max(1, int(args.jobs))
    failures = 0
    for i in range(0, len(specs), jobs):
        batch_idx = (i // jobs) + 1
        batch = specs[i : i + jobs]
        print(f"\n=== Batch {batch_idx} ({len(batch)} cmds) ===")
        codes = _run_batch(batch, repo_root=repo_root, env=env)
        if SHUTDOWN_REQUESTED:
            return 130
        batch_failures = sum(1 for c in codes if c != 0)
        failures += batch_failures
        if batch_failures and args.stop_on_failure:
            print("Stopping on failure.")
            return 1

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
