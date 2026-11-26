#!/usr/bin/env python3
"""Read-only training dashboard. Run over SSH with a PTY, or use --once."""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def events(path):
    try:
        with path.open() as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        yield item
                except ValueError:
                    pass  # A writer may not have finished its last line yet.
    except OSError:
        pass


def number(value):
    return f"{value:.4g}" if isinstance(value, (int, float)) else "--"


def render(run, pace=None, width=80, height=24, details=False, eval_only=False):
    cfg = read_json(run / "config.json")
    latest, workers, learning = {}, {}, {}
    completed = 0
    rewards = {}
    mode = "unknown"
    for event in events(run / "events.jsonl"):
        latest = event
        kind = event.get("kind")
        if kind == "run_started":
            mode = event.get("mode", "unknown")
        elif kind == "update":
            completed = max(completed, event.get("update", 0))
        elif kind == "stage":
            workers[event.get("worker", 0)] = event
            key = (event.get("update"), event.get("worker"))
            rewards.setdefault(key, {})[event.get("stage")] = event.get("reward")
        elif kind == "stage_update":
            learning[event.get("stage", "unknown")] = event
    total = cfg.get("updates", 0)
    fraction = min(completed / total, 1) if total else 0
    episode_rewards = [sum(stages.values()) for (update, _), stages in rewards.items()
                       if update == completed and all(isinstance(stages.get(stage), (float, int))
                       for stage in ("floorplanning", "placement", "cts", "routing"))]
    reward_text = number(sum(episode_rewards) / len(episode_rewards)) if episode_rewards else "waiting"
    eta = "measuring update speed..."
    speed = "--"
    if pace is not None:
        now = time.monotonic()
        if not pace or completed < pace[0][1]:
            pace[:] = [(now, completed)]
        elif completed > pace[-1][1]:
            pace.append((now, completed))
        if len(pace) > 1:
            seconds = (pace[-1][0] - pace[0][0]) / (pace[-1][1] - pace[0][1])
            remaining = max(0, (total - completed) * seconds - (now - pace[-1][0]))
            hours, remainder = divmod(int(remaining), 3600)
            minutes, secs = divmod(remainder, 60)
            eta = f"~{hours}h {minutes:02}m {secs:02}s" if total else "unknown target"
            speed = f"~{seconds:.0f}s/update"
    if latest.get("kind") == "run_finished":
        eta = "finished"
    bar = "#" * int(fraction * 30) + "-" * (30 - int(fraction * 30))
    try:
        age = max(0, int(time.time() - (run / "events.jsonl").stat().st_mtime))
        freshness = f"{age}s ago"
    except OSError:
        freshness = "waiting for event file"
    status = latest.get("status", "finished") if latest.get("kind") == "run_finished" else "Watching logs (process status unknown)"
    width = max(20, width - 1)
    rule = "─" * width
    bar_width = max(8, min(40, width - 12))
    filled = int(fraction * bar_width)
    bar = "━" * filled + "·" * (bar_width - filled)
    lines = [
        "MAPPO / TRAINING",
        rule,
        f"UPDATES    {completed} / {total or '?'}   ({fraction:.1%})",
        f"{bar}",
        f"TIME LEFT  {eta}",
        f"REWARD     {reward_text}  (mean episode)",
        f"PACE       {speed}",
        rule,
        f"Last log: {freshness}",
        f"Mode: {mode}",
    ]
    if latest.get("kind") == "run_finished":
        lines.append(f"Result: {status}")
    evaluation = read_json(run / "evaluation.json")
    if eval_only:
        lines = ["MAPPO / CHECKPOINT EVALUATION", rule]
    else:
        lines += [rule, "CHECKPOINT EVALUATION"]
    if evaluation:
        checkpoint_name = Path(evaluation.get("checkpoint", "unknown")).name
        lines.append(f"{checkpoint_name}: {evaluation.get('status', '?')}")
        if evaluation.get("status") == "complete":
            lines.append(f"Eval reward: {number(evaluation.get('mean_return'))}")
            episodes = evaluation.get("episodes", [])
            if isinstance(episodes, list) and episodes:
                metrics = episodes[-1].get("metrics", {})
                lines.append(f"Area: {number(metrics.get('area_um2'))} um2 | DRC: {number(metrics.get('drc_violations'))}")
                lines.append(f"Slack: {number(metrics.get('worst_slack_ns'))} ns (last eval episode)")
        elif evaluation.get("status") == "failed":
            lines.append(f"Error: {evaluation.get('error', 'see evaluation.json')}")
        else:
            lines.append(f"Episode {evaluation.get('episode', '?')}: {evaluation.get('stage', 'loading')}")
    else:
        lines.append("No evaluation recorded yet")
    if eval_only:
        lines += ["", "Evaluation runs on the desktop.", "Results refresh automatically."]
        footer = "Ctrl+C: close viewer"
        visible = lines[:max(1, height - 3)] + [rule, footer]
        return "\n".join(line[:width] for line in visible)
    lines.append("Latest reported worker results:")
    for worker, event in sorted(workers.items()):
        stage = event.get('stage', '?')
        reward = number(event.get('reward'))
        if width >= 65:
            lines.append(f"  #{worker}   update {event.get('update', '?'):<4}  {stage:<13} reward {reward}")
        else:
            lines.append(f"  #{worker} u{event.get('update', '?')} {stage}: {reward}")
    if not workers:
        lines.append("  Waiting for stage results...")
    checkpoints = sorted(run.glob("checkpoint-*.pt"))
    lines.append(f"Saved: {checkpoints[-1].name if checkpoints else 'no checkpoint yet'}")
    failures = list(events(run / "failures.jsonl"))
    if failures:
        lines.insert(8, f"ERROR: {failures[-1].get('error', 'see failures.jsonl')}")
    if details:
        lines += [rule, "POLICY LOSSES  (actor / value)"]
        for stage, event in learning.items():
            lines.append(f"{stage:<14} {number(event.get('actor_loss'))} / {number(event.get('value_loss'))}")
    footer = "Ctrl+C exits viewer; training continues."
    if width < len(footer):
        footer = "Ctrl+C: close viewer"
    # Reserve the bottom row and avoid wrapping at the terminal's right edge.
    lines = lines[:max(1, height - 3)] + [rule, footer]
    lines = [line if len(line) <= width else line[:width - 1] + "…" for line in lines]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--once", action="store_true", help="Print one snapshot without terminal controls")
    parser.add_argument("--interval", type=float, default=2)
    parser.add_argument("--details", action="store_true", help="Also show policy losses in taller terminals")
    parser.add_argument("--plain", action="store_true", help="Disable colors")
    parser.add_argument("--eval-only", action="store_true", help="Show only checkpoint evaluation")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    if not args.run_dir.is_dir():
        parser.error(f"Run directory does not exist: {args.run_dir}")
    if args.once or not sys.stdout.isatty():
        print(render(args.run_dir, details=args.details, eval_only=args.eval_only))
        return
    try:
        print("\033[?1049h\033[?25l", end="", flush=True)
        pace = []
        while True:
            width, height = shutil.get_terminal_size()
            lines = render(args.run_dir, pace, width, height, args.details, args.eval_only).splitlines()
            colored = []
            for index, line in enumerate(lines):
                color = ""
                if not args.plain:
                    color = "\033[1;36m" if index in (0, 2, 3) else "\033[1;32m" if index == 5 else "\033[0m"
                colored.append(color + line + "\033[0m\033[K")
            # Overwrite in place; clearing the entire screen every refresh flickers over SSH.
            print("\033[H" + "\r\n".join(colored) + "\033[J", end="", flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        print("\033[0m\033[?25h\033[?1049l", end="", flush=True)


if __name__ == "__main__":
    main()
