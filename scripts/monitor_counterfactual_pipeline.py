from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class RunInfo:
    pid: int
    started_local: datetime
    started_utc: datetime
    elapsed: str
    command: str
    model: str | None
    input_dir: Path | None
    input_ids: Path | None
    input_path: Path | None
    csv_path: Path | None
    max_attempts: int | None
    candidate_budget: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize active generate_counterfactual runs and their current progress.",
    )
    parser.add_argument(
        "--watch",
        type=float,
        default=0.0,
        help="Refresh every N seconds until interrupted.",
    )
    parser.add_argument(
        "--ps-snapshot",
        type=Path,
        default=None,
        help="Optional file containing `ps -Ao pid=,lstart=,etime=,command=` output.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_ps_line(line: str) -> RunInfo | None:
    parts = line.strip().split(None, 7)
    if len(parts) != 8:
        return None
    pid_s, dow, mon, day, clock, year, elapsed, command = parts
    if "scripts.generate_counterfactual" not in command:
        return None

    started_local = datetime.strptime(
        f"{dow} {mon} {day} {clock} {year}",
        "%a %b %d %H:%M:%S %Y",
    ).astimezone()
    started_utc = started_local.astimezone(timezone.utc)

    tokens = shlex.split(command)
    opts = parse_command_options(tokens)
    return RunInfo(
        pid=int(pid_s),
        started_local=started_local,
        started_utc=started_utc,
        elapsed=elapsed,
        command=command,
        model=opts.get("model"),
        input_dir=resolve_path(opts.get("input-dir")),
        input_ids=resolve_path(opts.get("input-ids")),
        input_path=resolve_path(opts.get("input-path")),
        csv_path=resolve_path(opts.get("csv-path")),
        max_attempts=parse_int(opts.get("max-attempts")),
        candidate_budget=parse_int(opts.get("candidate-budget")),
    )


def parse_command_options(tokens: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if not token.startswith("--"):
            idx += 1
            continue
        key = token[2:]
        if "=" in key:
            key, value = key.split("=", 1)
            result[key] = value
            idx += 1
            continue
        if idx + 1 < len(tokens) and not tokens[idx + 1].startswith("--"):
            result[key] = tokens[idx + 1]
            idx += 2
            continue
        result[key] = "true"
        idx += 1
    return result


def resolve_path(raw: str | None) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    return repo_root() / path


def parse_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def active_runs(ps_snapshot: Path | None = None) -> list[RunInfo]:
    if ps_snapshot:
        output = ps_snapshot.read_text(encoding="utf-8")
    else:
        cmd = ["ps", "-Ao", "pid=,lstart=,etime=,command="]
        output = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    runs = [run for line in output.splitlines() if (run := parse_ps_line(line))]
    return sorted(runs, key=lambda run: run.started_local)


def iter_images(path: Path) -> Iterable[Path]:
    for entry in sorted(path.iterdir()):
        if entry.is_file() and entry.suffix.lower() in IMAGE_SUFFIXES:
            yield entry


def input_count(run: RunInfo) -> int | None:
    if run.input_path:
        return 1
    if run.input_ids and run.input_ids.exists():
        return sum(1 for line in run.input_ids.read_text(encoding="utf-8").splitlines() if line.strip())
    if run.input_dir and run.input_dir.exists():
        return sum(1 for _ in iter_images(run.input_dir))
    return None


def infer_auto_csv(run: RunInfo) -> Path | None:
    eval_dir = repo_root() / "data/03_eval_results"
    if not eval_dir.exists():
        return None
    started_epoch = run.started_local.timestamp()
    candidates = []
    for path in eval_dir.glob("counterfactual_results_*.csv"):
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            continue
        if mtime + 2 < started_epoch:
            continue
        candidates.append((mtime, path))
    if not candidates:
        return None
    return max(candidates)[1]


def summarize_csv(csv_path: Path | None) -> dict[str, object]:
    summary = {
        "exists": False,
        "rows": 0,
        "processed_images": 0,
        "valid_rows": 0,
        "error_rows": 0,
        "mtime": None,
    }
    if not csv_path or not csv_path.exists():
        return summary

    processed_inputs: set[str] = set()
    valid_rows = 0
    error_rows = 0
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        image_path = (row.get("input_image_path") or "").strip()
        if image_path:
            processed_inputs.add(image_path)
        if str(row.get("critic_is_valid", "")).strip().lower() == "true":
            valid_rows += 1
        if str(row.get("critic_notes", "")).startswith("ERROR:"):
            error_rows += 1

    summary.update(
        {
            "exists": True,
            "rows": len(rows),
            "processed_images": len(processed_inputs),
            "valid_rows": valid_rows,
            "error_rows": error_rows,
            "mtime": datetime.fromtimestamp(csv_path.stat().st_mtime).astimezone(),
        }
    )
    return summary


def summarize_outputs(run: RunInfo) -> dict[str, object]:
    output_dir = repo_root() / "data/02_counterfactual"
    summary = {
        "count_since_start": 0,
        "latest_path": None,
        "latest_mtime": None,
    }
    if not output_dir.exists():
        return summary

    started_epoch = run.started_local.timestamp()
    recent: list[tuple[float, Path]] = []
    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        if stat.st_mtime >= started_epoch:
            recent.append((stat.st_mtime, path))
    if not recent:
        return summary

    latest_mtime, latest_path = max(recent)
    summary.update(
        {
            "count_since_start": len(recent),
            "latest_path": latest_path,
            "latest_mtime": datetime.fromtimestamp(latest_mtime).astimezone(),
        }
    )
    return summary


def format_percent(done: int, total: int | None) -> str:
    if not total:
        return "n/a"
    return f"{(100.0 * done / total):.1f}%"


def status_hint(csv_summary: dict[str, object], output_summary: dict[str, object]) -> str:
    if not csv_summary["exists"] and output_summary["count_since_start"]:
        return "Active generation, but no image-level checkpoint has been written yet."
    if csv_summary["exists"] and output_summary["count_since_start"]:
        csv_mtime = csv_summary["mtime"]
        latest_output_mtime = output_summary["latest_mtime"]
        if isinstance(csv_mtime, datetime) and isinstance(latest_output_mtime, datetime):
            if latest_output_mtime > csv_mtime:
                return "Generating within the current image; CSV updates only after an image completes."
    if csv_summary["error_rows"] and not output_summary["count_since_start"]:
        return "Only error rows are visible so far; inspect the model and planner settings."
    return "No obvious stall signal from filesystem activity."


def render_run(run: RunInfo) -> str:
    total_inputs = input_count(run)
    csv_path = run.csv_path if run.csv_path else infer_auto_csv(run)
    csv_summary = summarize_csv(csv_path)
    output_summary = summarize_outputs(run)
    processed_images = int(csv_summary["processed_images"])

    lines = [
        f"PID {run.pid} | elapsed={run.elapsed} | started={run.started_local.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"model={run.model or 'default'} | max_attempts={run.max_attempts or 'default'} | candidate_budget={run.candidate_budget or 'default'}",
        f"input_dir={display_path(run.input_dir)}",
        f"input_ids={display_path(run.input_ids)} | input_total={total_inputs if total_inputs is not None else 'unknown'}",
        f"csv={display_path(csv_path)}",
    ]

    if csv_summary["exists"]:
        mtime = csv_summary["mtime"]
        lines.append(
            "csv_progress="
            f"{processed_images}/{total_inputs if total_inputs is not None else '?'} images "
            f"({format_percent(processed_images, total_inputs)}) | "
            f"{csv_summary['rows']} rows | {csv_summary['valid_rows']} valid | {csv_summary['error_rows']} errors | "
            f"updated={mtime.strftime('%Y-%m-%d %H:%M:%S %Z') if isinstance(mtime, datetime) else 'unknown'}"
        )
    else:
        lines.append("csv_progress=no checkpoint CSV written for this run yet")

    latest_path = output_summary["latest_path"]
    latest_mtime = output_summary["latest_mtime"]
    lines.append(
        "outputs_since_start="
        f"{output_summary['count_since_start']} | latest_output="
        f"{display_path(latest_path)}"
        + (
            f" @ {latest_mtime.strftime('%Y-%m-%d %H:%M:%S %Z')}"
            if isinstance(latest_mtime, datetime)
            else ""
        )
    )
    lines.append(f"hint={status_hint(csv_summary, output_summary)}")
    lines.append(f"command={run.command}")
    return "\n".join(lines)


def display_path(path: Path | None) -> str:
    if not path:
        return "n/a"
    try:
        return str(path.relative_to(repo_root()))
    except ValueError:
        return str(path)


def render_all(ps_snapshot: Path | None = None) -> str:
    runs = active_runs(ps_snapshot=ps_snapshot)
    header = f"Counterfactual pipeline monitor | {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}"
    if not runs:
        return f"{header}\n\nNo active scripts.generate_counterfactual processes found."
    blocks = [header]
    for run in runs:
        blocks.append("")
        blocks.append(render_run(run))
    return "\n".join(blocks)


def main() -> None:
    args = parse_args()
    if args.watch and args.watch > 0:
        try:
            while True:
                print("\033[2J\033[H", end="")
                print(render_all(ps_snapshot=args.ps_snapshot))
                time.sleep(args.watch)
        except KeyboardInterrupt:
            return
    else:
        print(render_all(ps_snapshot=args.ps_snapshot))


if __name__ == "__main__":
    sys.exit(main())
