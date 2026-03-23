#!/usr/bin/env python3
"""Run Treba on a PAutomaC problem and evaluate perplexity.

This script is a Python replacement for the historical `pautomac.perl` wrapper.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import tarfile
from typing import Iterable

from treba_wrapper import TrebaRunner


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _find_member(tar: tarfile.TarFile, suffix: str) -> tarfile.TarInfo:
    matches: list[tarfile.TarInfo] = []
    for member in tar.getmembers():
        if not member.isfile():
            continue
        name = member.name
        base = name.rsplit("/", 1)[-1]
        if base.startswith("._"):
            continue
        if name.endswith(suffix):
            matches.append(member)

    if not matches:
        raise FileNotFoundError(f"Could not find member ending in {suffix!r} in tar archive")

    # Prefer shortest path if multiple candidates (e.g., nested root folder).
    matches.sort(key=lambda m: len(m.name))
    return matches[0]


def _read_problem_files(dataset_path: Path, problem: int) -> tuple[str, str, str]:
    train_name = f"{problem}.pautomac.train"
    test_name = f"{problem}.pautomac.test"
    sol_name = f"{problem}.pautomac_solution.txt"

    if dataset_path.is_dir():
        train = _read_text(dataset_path / train_name)
        test = _read_text(dataset_path / test_name)
        sol = _read_text(dataset_path / sol_name)
        return train, test, sol

    if dataset_path.is_file() and tarfile.is_tarfile(dataset_path):
        with tarfile.open(dataset_path, "r:*") as tar:
            train_member = _find_member(tar, f"/{train_name}")
            test_member = _find_member(tar, f"/{test_name}")
            sol_member = _find_member(tar, f"/{sol_name}")

            train_b = tar.extractfile(train_member)
            test_b = tar.extractfile(test_member)
            sol_b = tar.extractfile(sol_member)
            if train_b is None or test_b is None or sol_b is None:
                raise RuntimeError("Failed to extract one or more PAutomaC files")

            train = train_b.read().decode("utf-8")
            test = test_b.read().decode("utf-8")
            sol = sol_b.read().decode("utf-8")
            return train, test, sol

    raise FileNotFoundError(
        f"Dataset path must be a directory or tar archive: {dataset_path}"
    )


def _convert_pautomac_to_treba(raw_text: str, use_hmm: bool) -> tuple[str, int]:
    lines = raw_text.splitlines()
    if not lines:
        raise ValueError("Empty PAutomaC file")

    header = lines[0].split()
    if len(header) < 2:
        raise ValueError(f"Invalid PAutomaC header: {lines[0]!r}")
    alphabet_size = int(header[1])

    output_lines: list[str] = []
    for line in lines[1:]:
        parts = line.split()
        if not parts:
            output_lines.append("")
            continue

        if use_hmm:
            converted = parts[1:]
        else:
            converted = [str(alphabet_size), *parts[1:]]

        output_lines.append(" ".join(converted))

    return "\n".join(output_lines) + "\n", alphabet_size


def _parse_float_lines(text: str) -> list[float]:
    values: list[float] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped:
            values.append(float(stripped))
    return values


def _normalize_from_log10(log10_values: Iterable[float], legacy: bool) -> list[float]:
    values = list(log10_values)
    if not values:
        raise ValueError("No likelihood values were produced")

    if legacy:
        # Match historical perl wrapper behavior exactly.
        shift = 0.0
        for value in values:
            if value < abs(shift):
                shift = value
    else:
        # Numerically stable log-sum-exp equivalent in base-10.
        shift = max(values)

    lsum = math.log10(sum(10 ** (v - shift) for v in values)) + shift
    return [10 ** (v - lsum) for v in values]


def _compute_perplexity(predictions: list[float], gold: list[float]) -> tuple[float, float]:
    if len(predictions) != len(gold):
        raise ValueError(
            f"Prediction/gold length mismatch: {len(predictions)} vs {len(gold)}"
        )

    score = 0.0
    min_score = 0.0
    tiny = sys.float_info.min

    for g, p in zip(gold, predictions, strict=True):
        if g == 0.0:
            continue
        p = max(p, tiny)
        score += g * math.log2(p)
        min_score += g * math.log2(g)

    perplexity = 2 ** (-score)
    min_perplexity = 2 ** (-min_score)
    return perplexity, min_perplexity


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Treba on PAutomaC data and evaluate perplexity"
    )
    parser.add_argument("dataset", help="PAutomaC directory or tar archive")
    parser.add_argument("problem", type=int, help="Problem number (1..48)")
    parser.add_argument(
        "treba_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed directly to treba (e.g. --train=merge --alpha=0.3)",
    )
    parser.add_argument(
        "--treba-bin",
        default="treba",
        help="Path/name of treba binary (default: treba, falls back to ./treba)",
    )
    parser.add_argument(
        "--work-dir",
        default="runs/pautomac",
        help="Directory for generated artifacts",
    )
    parser.add_argument(
        "--legacy-normalizer",
        action="store_true",
        help="Use exact historical perl normalization shift logic",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON result",
    )
    args = parser.parse_args()

    treba_args = list(args.treba_args)
    if treba_args and treba_args[0] == "--":
        treba_args = treba_args[1:]

    if not treba_args:
        parser.error("No Treba args provided; pass training args after the problem number")

    if not any(a.startswith("--train=") for a in treba_args):
        parser.error("Treba args must include a training mode (e.g. --train=merge)")

    use_hmm = any(a == "--hmm" or a.startswith("--hmm=") for a in treba_args)
    machine_suffix = "hmm" if use_hmm else "fsa"

    dataset = Path(args.dataset)
    train_raw, test_raw, sol_raw = _read_problem_files(dataset, args.problem)

    train_obs, _alphabet_size = _convert_pautomac_to_treba(train_raw, use_hmm=use_hmm)
    test_obs, _ = _convert_pautomac_to_treba(test_raw, use_hmm=use_hmm)

    artifact_dir = Path(args.work_dir) / f"problem_{args.problem:02d}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train_obs_path = artifact_dir / f"{args.problem}.train.stripped.obs"
    test_obs_path = artifact_dir / f"{args.problem}.test.stripped.obs"
    model_path = artifact_dir / f"{args.problem}.pautomac.{machine_suffix}"
    pred_log10_path = artifact_dir / f"{args.problem}.pred.log10"
    pred_prob_path = artifact_dir / f"{args.problem}.pred.prob"

    train_obs_path.write_text(train_obs, encoding="utf-8")
    test_obs_path.write_text(test_obs, encoding="utf-8")

    runner = TrebaRunner(args.treba_bin)

    train_result = runner.run([*treba_args, str(train_obs_path)])
    if train_result.stderr:
        sys.stderr.write(train_result.stderr)
    if train_result.returncode != 0:
        sys.stderr.write(
            f"Training failed (exit={train_result.returncode}): {' '.join(train_result.cmd)}\n"
        )
        return train_result.returncode

    model_path.write_text(train_result.stdout, encoding="utf-8")

    likelihood_args: list[str] = [
        "--output-format=log10",
        "--likelihood=f",
        f"--file={model_path}",
        str(test_obs_path),
    ]
    if use_hmm:
        likelihood_args.insert(0, "--hmm")

    likelihood_result = runner.run(likelihood_args)
    if likelihood_result.stderr:
        sys.stderr.write(likelihood_result.stderr)
    if likelihood_result.returncode != 0:
        sys.stderr.write(
            f"Likelihood failed (exit={likelihood_result.returncode}): {' '.join(likelihood_result.cmd)}\n"
        )
        return likelihood_result.returncode

    pred_log10_path.write_text(likelihood_result.stdout, encoding="utf-8")

    log10_values = _parse_float_lines(likelihood_result.stdout)
    probabilities = _normalize_from_log10(log10_values, legacy=args.legacy_normalizer)
    pred_prob_path.write_text("\n".join(f"{p:.18g}" for p in probabilities) + "\n", encoding="utf-8")

    sol_lines = [line.strip() for line in sol_raw.splitlines() if line.strip()]
    if not sol_lines:
        raise ValueError("Empty solution file")
    gold = [float(x) for x in sol_lines[1:]]

    perplexity, min_perplexity = _compute_perplexity(probabilities, gold)

    result = {
        "problem": args.problem,
        "use_hmm": use_hmm,
        "perplexity": perplexity,
        "minimal_perplexity": min_perplexity,
        "treba_args": treba_args,
        "artifacts": {
            "artifact_dir": str(artifact_dir),
            "train_obs": str(train_obs_path),
            "test_obs": str(test_obs_path),
            "model": str(model_path),
            "pred_log10": str(pred_log10_path),
            "pred_prob": str(pred_prob_path),
        },
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"Minimal perplexity (solution): {min_perplexity}")
        print(f"Perplexity: {perplexity}")
        print(f"Artifacts: {artifact_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
