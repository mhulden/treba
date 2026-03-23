#!/usr/bin/env python3
"""Run a configurable battery of PAutomaC experiments for Treba.

This orchestrates many calls to scripts/pautomac_eval.py, stores per-run artifacts,
and writes machine-readable + human-readable summaries.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    description: str
    treba_args: list[str]
    stochastic: bool
    requires_cuda_device: bool


@dataclass(frozen=True)
class Task:
    model: ModelSpec
    problem: int


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def _load_presets(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _split_csv_args(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        for piece in value.split(","):
            piece = piece.strip()
            if piece:
                out.append(piece)
    return out


def _parse_problem_spec(spec: str) -> list[int]:
    spec = spec.strip().lower()
    if spec == "all":
        return list(range(1, 49))

    problems: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left)
            end = int(right)
            if start > end:
                start, end = end, start
            problems.update(range(start, end + 1))
        else:
            problems.add(int(part))

    if not problems:
        raise ValueError("No problems selected")

    invalid = sorted(p for p in problems if p < 1 or p > 48)
    if invalid:
        raise ValueError(f"Problem IDs must be in [1,48], got: {invalid}")

    return sorted(problems)


def _extract_json(text: str) -> dict[str, Any]:
    payload = text.strip()
    if not payload:
        raise ValueError("Empty JSON output")

    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        # Fallback for unexpected chatter before/after JSON payload.
        start = payload.find("{")
        end = payload.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise
        return json.loads(payload[start : end + 1])


def _detect_cuda_capability(treba_bin: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "treba_has_cuda_support": False,
        "nvidia_smi_present": False,
        "nvidia_gpu_visible": False,
        "cuda_usable": False,
        "treba_version_output": "",
        "nvidia_smi_output": "",
    }

    try:
        proc = subprocess.run(
            [treba_bin, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        out = (proc.stdout or "").strip()
        info["treba_version_output"] = out
        info["treba_has_cuda_support"] = "compiled with CUDA support" in out
    except OSError as exc:
        info["treba_version_output"] = f"error: {exc}"

    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        info["nvidia_smi_output"] = (proc.stdout or "").strip()
        info["nvidia_smi_present"] = proc.returncode == 0 or bool(proc.stdout)
        info["nvidia_gpu_visible"] = proc.returncode == 0 and bool(proc.stdout.strip())
    except OSError:
        info["nvidia_smi_output"] = "nvidia-smi not available"

    info["cuda_usable"] = bool(
        info["treba_has_cuda_support"] and info["nvidia_gpu_visible"]
    )
    return info


def _select_models(
    presets: dict[str, Any],
    group: str,
    selected_models: list[str] | None,
) -> list[ModelSpec]:
    model_map: dict[str, Any] = presets["models"]

    if selected_models:
        model_ids = selected_models
    else:
        groups = presets.get("groups", {})
        if group not in groups:
            raise ValueError(f"Unknown group: {group}")
        model_ids = list(groups[group])

    specs: list[ModelSpec] = []
    for model_id in model_ids:
        if model_id not in model_map:
            raise ValueError(f"Unknown model id in presets: {model_id}")
        cfg = model_map[model_id]
        specs.append(
            ModelSpec(
                model_id=model_id,
                description=cfg.get("description", model_id),
                treba_args=list(cfg["treba_args"]),
                stochastic=bool(cfg.get("stochastic", True)),
                requires_cuda_device=bool(cfg.get("requires_cuda_device", False)),
            )
        )
    return specs


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_results_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")


def _write_results_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "run_id",
        "timestamp_utc",
        "model_id",
        "problem",
        "status",
        "returncode",
        "duration_sec",
        "perplexity",
        "minimal_perplexity",
        "stochastic",
        "requires_cuda_device",
        "artifact_dir",
        "command",
        "error",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k, "") for k in fields})


def _write_summary(path: Path, records: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_model.setdefault(rec["model_id"], []).append(rec)

    lines: list[str] = []
    lines.append(f"# PAutomaC Battery Summary ({manifest['run_id']})")
    lines.append("")
    lines.append(f"- Started (UTC): {manifest['started_utc']}")
    lines.append(f"- Finished (UTC): {manifest['finished_utc']}")
    lines.append(f"- Dataset: `{manifest['dataset']}`")
    lines.append(f"- Problems: `{manifest['problems']}`")
    lines.append(f"- Models: `{', '.join(manifest['model_ids'])}`")
    lines.append("")
    lines.append("## Per-Model")
    lines.append("")

    for model_id in manifest["model_ids"]:
        rows = by_model.get(model_id, [])
        ok = [r for r in rows if r["status"] == "ok"]
        fail = [r for r in rows if r["status"] == "failed"]
        skip = [r for r in rows if r["status"] == "skipped"]

        lines.append(f"### {model_id}")
        lines.append(f"- ok: {len(ok)}")
        lines.append(f"- failed: {len(fail)}")
        lines.append(f"- skipped: {len(skip)}")
        if ok:
            perplexities = [float(r["perplexity"]) for r in ok]
            lines.append(f"- perplexity mean: {statistics.mean(perplexities):.12g}")
            lines.append(f"- perplexity median: {statistics.median(perplexities):.12g}")
            lines.append(f"- perplexity min: {min(perplexities):.12g}")
            lines.append(f"- perplexity max: {max(perplexities):.12g}")
        lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _record_base(run_id: str, model: ModelSpec, problem: int) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model.model_id,
        "problem": problem,
        "stochastic": model.stochastic,
        "requires_cuda_device": model.requires_cuda_device,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PAutomaC battery experiments")
    parser.add_argument("dataset", nargs="?", help="PAutomaC dir or tar archive (required unless --list-models)")
    parser.add_argument("--treba-bin", default="./treba", help="Treba binary path/name")
    parser.add_argument("--eval-script", default="scripts/pautomac_eval.py", help="Path to pautomac_eval.py")
    parser.add_argument(
        "--preset-file",
        default="scripts/pautomac_battery_presets.json",
        help="Preset JSON file",
    )
    parser.add_argument("--group", default="all_models_v1", help="Preset group name")
    parser.add_argument("--models", nargs="*", help="Specific model IDs (space or comma separated)")
    parser.add_argument("--problems", default="1-48", help="Problem spec, e.g. 1-48 or 1,2,14")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel jobs")
    parser.add_argument("--work-dir", default="runs/pautomac_battery", help="Output root")
    parser.add_argument("--run-id", default=None, help="Optional run id")
    parser.add_argument("--cuda-mode", choices=["auto", "off", "force"], default="auto")
    parser.add_argument("--legacy-normalizer", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Plan only; do not execute tasks")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    parser.add_argument("--keep-going", action="store_true", help="Continue when individual tasks fail")

    args = parser.parse_args()

    preset_file = Path(args.preset_file)
    presets = _load_presets(preset_file)

    selected_models_raw = _split_csv_args(args.models or [])
    models = _select_models(
        presets=presets,
        group=args.group,
        selected_models=selected_models_raw if selected_models_raw else None,
    )

    if args.list_models:
        for model in models:
            cuda_tag = " [cuda]" if model.requires_cuda_device else ""
            stochastic_tag = "stochastic" if model.stochastic else "deterministic"
            print(f"{model.model_id}: {model.description}{cuda_tag} ({stochastic_tag})")
            print(f"  args: {' '.join(model.treba_args)}")
        return 0

    if not args.dataset:
        parser.error("dataset is required unless --list-models is used")

    dataset = Path(args.dataset)
    problems = _parse_problem_spec(args.problems)

    run_id = args.run_id or _timestamp()
    run_root = Path(args.work_dir) / run_id
    logs_dir = run_root / "logs"
    artifacts_root = run_root / "artifacts"
    logs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_root.mkdir(parents=True, exist_ok=True)

    cuda_info = _detect_cuda_capability(args.treba_bin)

    tasks: list[Task] = []
    records: list[dict[str, Any]] = []

    for model in models:
        should_skip_cuda = False
        skip_reason = ""
        if model.requires_cuda_device:
            if args.cuda_mode == "off":
                should_skip_cuda = True
                skip_reason = "cuda-mode=off"
            elif args.cuda_mode == "auto" and not cuda_info["cuda_usable"]:
                should_skip_cuda = True
                skip_reason = "cuda unavailable (binary not CUDA-capable or no visible NVIDIA GPU)"

        if should_skip_cuda:
            for problem in problems:
                rec = _record_base(run_id, model, problem)
                rec.update(
                    {
                        "status": "skipped",
                        "returncode": "",
                        "duration_sec": 0.0,
                        "perplexity": "",
                        "minimal_perplexity": "",
                        "artifact_dir": "",
                        "command": "",
                        "error": skip_reason,
                    }
                )
                records.append(rec)
            continue

        for problem in problems:
            tasks.append(Task(model=model, problem=problem))

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "finished_utc": "",
        "dataset": str(dataset),
        "problems": args.problems,
        "jobs": args.jobs,
        "group": args.group,
        "model_ids": [m.model_id for m in models],
        "cuda_mode": args.cuda_mode,
        "legacy_normalizer": args.legacy_normalizer,
        "treba_bin": args.treba_bin,
        "eval_script": args.eval_script,
        "preset_file": str(preset_file),
        "cuda_detection": cuda_info,
        "task_count": len(tasks),
        "skipped_count": len([r for r in records if r["status"] == "skipped"]),
    }
    _write_manifest(run_root / "manifest.json", manifest)

    if args.dry_run:
        plan = {
            "run_id": run_id,
            "tasks": [
                {
                    "model_id": t.model.model_id,
                    "problem": t.problem,
                    "treba_args": t.model.treba_args,
                }
                for t in tasks
            ],
            "skipped": records,
        }
        (run_root / "plan.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"Dry run complete: {run_root}")
        print(f"Planned tasks: {len(tasks)}")
        print(f"Pre-skipped tasks: {len(records)}")
        return 0

    eval_script = Path(args.eval_script)

    def run_task(task: Task) -> dict[str, Any]:
        start = time.perf_counter()
        model = task.model
        base = _record_base(run_id, model, task.problem)

        model_artifact_root = artifacts_root / model.model_id
        model_artifact_root.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(eval_script),
            "--json",
            "--treba-bin",
            args.treba_bin,
            "--work-dir",
            str(model_artifact_root),
        ]
        if args.legacy_normalizer:
            cmd.append("--legacy-normalizer")
        cmd.extend([str(dataset), str(task.problem), *model.treba_args])

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        elapsed = time.perf_counter() - start

        stdout_log = logs_dir / f"{model.model_id}_p{task.problem:02d}.stdout.log"
        stderr_log = logs_dir / f"{model.model_id}_p{task.problem:02d}.stderr.log"
        stdout_log.write_text(proc.stdout or "", encoding="utf-8")
        stderr_log.write_text(proc.stderr or "", encoding="utf-8")

        record = dict(base)
        record["command"] = shlex.join(cmd)
        record["returncode"] = proc.returncode
        record["duration_sec"] = round(elapsed, 6)

        if proc.returncode != 0:
            record.update(
                {
                    "status": "failed",
                    "perplexity": "",
                    "minimal_perplexity": "",
                    "artifact_dir": "",
                    "error": f"exit={proc.returncode}",
                }
            )
            return record

        try:
            payload = _extract_json(proc.stdout)
            record.update(
                {
                    "status": "ok",
                    "perplexity": payload.get("perplexity", ""),
                    "minimal_perplexity": payload.get("minimal_perplexity", ""),
                    "artifact_dir": payload.get("artifacts", {}).get("artifact_dir", ""),
                    "error": "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            record.update(
                {
                    "status": "failed",
                    "perplexity": "",
                    "minimal_perplexity": "",
                    "artifact_dir": "",
                    "error": f"json-parse-error: {exc}",
                }
            )
        return record

    failures = 0

    if args.jobs <= 1:
        for idx, task in enumerate(tasks, start=1):
            print(f"[{idx}/{len(tasks)}] {task.model.model_id} problem {task.problem}")
            rec = run_task(task)
            records.append(rec)
            if rec["status"] == "failed":
                failures += 1
                if not args.keep_going:
                    print("Stopping on first failure (use --keep-going to continue).")
                    break
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            future_map = {pool.submit(run_task, task): task for task in tasks}
            done_count = 0
            for fut in as_completed(future_map):
                done_count += 1
                task = future_map[fut]
                rec = fut.result()
                print(
                    f"[{done_count}/{len(tasks)}] {task.model.model_id} problem {task.problem} -> {rec['status']}"
                )
                records.append(rec)
                if rec["status"] == "failed":
                    failures += 1

    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["ok_count"] = len([r for r in records if r["status"] == "ok"])
    manifest["failed_count"] = len([r for r in records if r["status"] == "failed"])
    manifest["skipped_count"] = len([r for r in records if r["status"] == "skipped"])

    _write_manifest(run_root / "manifest.json", manifest)
    _write_results_jsonl(run_root / "results.jsonl", records)
    _write_results_csv(run_root / "results.csv", records)
    _write_summary(run_root / "summary.md", records, manifest)

    print(f"Run complete: {run_root}")
    print(f"ok={manifest['ok_count']} failed={manifest['failed_count']} skipped={manifest['skipped_count']}")

    if failures and not args.keep_going:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
