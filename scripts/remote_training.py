#!/usr/bin/env python3
"""Mac/SSH client for the desktop dashboard and evaluated GDSII artifacts."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "remote.local.json"


def read_config(path):
    cfg = json.loads(path.read_text())
    for key in ("host", "remote_repo", "run_dir", "remote_python", "download_dir", "layout_app"):
        if not isinstance(cfg.get(key), str) or not cfg[key].strip():
            raise ValueError(f"Missing configuration value: {key}")
    if cfg["host"].startswith("-") or any(c.isspace() for c in cfg["host"]):
        raise ValueError("host must be an SSH alias or user@host, without options")
    if not PurePosixPath(cfg["remote_repo"]).is_absolute():
        raise ValueError("remote_repo must be an absolute desktop path")
    return cfg


def ssh_command(cfg, argv, terminal=False):
    command = "cd " + shlex.quote(cfg["remote_repo"]) + " && " + shlex.join(argv)
    return ["ssh", *(["-t"] if terminal else []), cfg["host"], command]


def dashboard(cfg, evaluation=False, once=False):
    argv = [cfg["remote_python"], "scripts/training_dashboard.py", cfg["run_dir"]]
    if evaluation:
        argv.append("--eval-only")
    if once:
        argv.append("--once")
    subprocess.run(ssh_command(cfg, argv, terminal=not once), check=True)


def download(cfg):
    # Resolve the GDS from the completed evaluation, not the best training layout.
    code = '''import json, sys
from pathlib import Path
run = Path(sys.argv[1]).resolve()
result = json.loads((run / 'evaluation.json').read_text())
if result.get('status') != 'complete' or result.get('mode') != 'real_openroad':
    raise SystemExit('No completed real evaluation is available yet')
episodes = result.get('episodes', [])
if not episodes:
    raise SystemExit('Evaluation has no episodes')
gds = (Path(episodes[-1]['stages'][-1]['workdir']) / 'results/nangate45/gcd/base/6_final.gds').resolve()
if not gds.is_relative_to(run) or not gds.is_file() or gds.stat().st_size == 0:
    raise SystemExit('Evaluated GDS is missing or outside the run directory')
print(json.dumps({'evaluation': result, 'gds': str(gds)}))
'''
    output = subprocess.check_output(ssh_command(cfg, [cfg["remote_python"], "-c", code, cfg["run_dir"]]), text=True)
    remote = json.loads(output)
    result = remote["evaluation"]
    name = PurePosixPath(result["checkpoint"]).stem
    if not name.startswith("checkpoint-") or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in name):
        raise ValueError("Unexpected checkpoint filename")
    folder = Path(cfg["download_dir"]).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{name}-evaluated.gds"
    with tempfile.NamedTemporaryFile(dir=folder, suffix=".part", delete=False) as stream:
        partial = Path(stream.name)
        try:
            subprocess.run(ssh_command(cfg, ["cat", "--", remote["gds"]]), stdout=stream, check=True)
            stream.flush()
            if not stream.tell():
                raise ValueError("Downloaded layout is empty")
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)
    (folder / f"{name}-evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    (folder / "latest-layout.json").write_text(json.dumps({"file": target.name}) + "\n")
    print(f"Downloaded: {target}\nEvaluation report: {folder / (name + '-evaluation.json')}")
    return target


def open_layout(cfg, target=None):
    if sys.platform != "darwin":
        raise ValueError("The open command is for macOS; use your local GDS viewer on this platform")
    if target is None:
        folder = Path(cfg["download_dir"]).expanduser()
        name = json.loads((folder / "latest-layout.json").read_text())["file"]
        target = folder / Path(name).name
    if not target.is_file():
        raise ValueError("Layout not found; run download first")
    subprocess.run(["open", "-a", cfg["layout_app"], str(target)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Create a private local configuration")
    init.add_argument("--host", required=True)
    init.add_argument("--repo", required=True)
    init.add_argument("--run-dir", required=True)
    for name in ("dashboard", "eval"):
        view = commands.add_parser(name)
        view.add_argument("--once", action="store_true")
    fetch = commands.add_parser("download", help="Download the last episode of the completed evaluation and its report")
    fetch.add_argument("--open", action="store_true")
    commands.add_parser("open", help="Open the most recently downloaded layout on this Mac")
    args = parser.parse_args()
    if args.command == "init":
        cfg = json.loads((ROOT / "configs/remote.example.json").read_text())
        cfg.update(host=args.host, remote_repo=args.repo, run_dir=args.run_dir)
        args.config.parent.mkdir(parents=True, exist_ok=True)
        with args.config.open("x") as stream:
            json.dump(cfg, stream, indent=2)
            stream.write("\n")
        os.chmod(args.config, 0o600)
        print(f"Created {args.config}. Edit this file to change settings. SSH handles authentication.")
        return
    cfg = read_config(args.config)
    if args.command in ("dashboard", "eval"):
        dashboard(cfg, evaluation=args.command == "eval", once=args.once)
    elif args.command == "download":
        target = download(cfg)
        if args.open:
            open_layout(cfg, target)
    elif args.command == "open":
        open_layout(cfg)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
