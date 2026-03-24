---
name: treba-revival
description: Operate, extend, and maintain the revived Treba PFSA/HMM toolkit (C + optional CUDA), including the Python wrapper, notebook workflows, and PAutomaC evaluation/battery tooling.
---

# Treba Revival Skill

Use this skill when working in this revived `treba` repository.

This repo is no longer just legacy C code: it now includes a working Python package (`treba_py`), notebook examples, and reproducible PAutomaC evaluation/battery scripts.

## Current State (Snapshot)

- Core `treba` CLI builds on modern Linux and macOS.
- Optional CUDA build path compiles (`make CUDA=1`) when toolkit is present.
- Python wrapper package `treba_py` is implemented and usable (fit/score/decode/sample/draw/save/load).
- PAutomaC tooling is Python-first (`scripts/pautomac_eval.py`, `scripts/pautomac_battery.py`).
- Notebook examples exist in `notebooks/` as both `.py` (jupytext source) and `.ipynb`.

## Repo Map (What To Open First)

Core C/CUDA:

- `treba.c`: main CLI entrypoint, training/decoding/generation orchestration, threading.
- `gibbs.c`: CPU collapsed Gibbs samplers.
- `treba_cuda.cu`: CUDA Gibbs kernels and host integration.
- `dffa.c`: deterministic automata induction / merge-style training (`merge`, `mdi`).
- `observations.c`: parsing/sorting/dedup of observation sequences.
- `io.c`: model I/O and probability-format conversions.
- `fastlogexp.h`: numerics helpers used by core algorithms (fast log-add support, digamma, chi/binomial helpers).
- `man/treba.1`: authoritative CLI semantics.

Build/system:

- `Makefile`: modernized build logic, GSL discovery, optional CUDA flags.
- `INSTALL`: legacy install notes.

Python + docs:

- `treba_py/`: Python API package (`HMM`, `PFSA`, encoder, runner, model parsing, Graphviz draw).
- `pyproject.toml`: editable install metadata for the package.
- `docs/PYTHON_API_DRAFT.md`: API behavior reference.
- `README.md`: main project docs (logo, build notes, wrappers, battery flow).

PAutomaC tooling:

- `scripts/treba_wrapper.py`: minimal subprocess wrapper for scripts.
- `scripts/pautomac_eval.py`: single-problem PAutomaC evaluator (perl replacement).
- `scripts/pautomac_battery.py`: multi-model/multi-problem orchestration.
- `scripts/pautomac_battery_presets.json`: model presets/groups.
- `pautomac_final.tar.gz`: local PAutomaC dataset archive.

Notebooks:

- `notebooks/01_treba_py_quickstart.py` + `.ipynb`
- `notebooks/02_pautomac_mini_workflow.py` + `.ipynb`

## Build Matrix (Linux + macOS + CUDA)

Requirements:

- C toolchain (`gcc`/compatible), pthreads.
- GSL headers/libs.
- Optional CUDA toolkit for `CUDA=1` builds.

Standard commands:

```bash
make clean && make
make clean && make CUDA=1
```

Important behavior:

- Always `make clean` when switching CPU <-> CUDA builds.
- Makefile includes `-fcommon` for modern GCC compatibility with legacy global definitions.

GSL discovery logic in `Makefile`:

- macOS (`Darwin`): tries Homebrew prefix autodetection (`/opt/homebrew` then `/usr/local`).
- Other platforms: tries `pkg-config gsl`; falls back to `-lgsl -lgslcblas`.

Useful overrides:

```bash
make GSL_PREFIX="$(brew --prefix)"
make GSL_CFLAGS="-I/your/prefix/include" GSL_LIBS="-L/your/prefix/lib -lgsl -lgslcblas"
```

CUDA notes:

- Default arch flags: `sm_60 sm_70 sm_75 sm_80 sm_86`.
- `aarch64` path includes NVCC compatibility defines:
  - `-D__GNUC__=8 -D__GNUC_MINOR__=0`
- A GPU is not required to compile CUDA, but is required to run `--cuda` training.

## Python API (`treba_py`) Guidance

Primary classes:

- `PFSA(...)`
- `HMM(...)`

Core methods:

- `fit`, `score`, `predict_proba`, `decode`, `sample`, `draw`, `save`
- `PFSA.from_file(...)`, `HMM.from_file(...)`

Token UX highlights:

- Accepts token sequences, strings, file paths.
- Auto maps arbitrary hashable tokens to integer IDs for Treba.
- Unknown handling via `TokenizationConfig.unknown_policy`:
  - `error`
  - `ignore_sequence`
  - `use_unk`

Important caveat:

- If using `use_unk`, the unknown token must survive into the trained model alphabet.
- If it is dropped during training, wrapper raises `AlphabetError` before score/decode.

Runner behavior:

- `treba_py.runner.TrebaRunner` resolves binaries lazily and can discover local `treba` binaries up parent directories.
- This prevents common notebook CWD issues when loading models via `from_file(...)`.

Graphviz draw behavior:

- `draw()` returns `graphviz.Digraph`.
- Parallel edges are merged by default (same source/target), labels stacked with newline separators.
- Defaults tuned for notebook readability (smaller node/font sizing, compact graph sizing).

Install for local Python work:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m pip install graphviz
```

## Notebook Workflow

Jupytext source-of-truth:

- Keep `.py` notebooks authoritative.
- Regenerate `.ipynb` via:

```bash
jupytext --to ipynb notebooks/*.py
```

Local noise to avoid committing:

- `notebooks/.ipynb_checkpoints/`
- `notebooks/artifacts/`

These often appear after notebook runs; clean before commit if present.

## PAutomaC Workflow

Single-problem evaluation (perl replacement):

```bash
python3 scripts/pautomac_eval.py --treba-bin ./treba pautomac_final.tar.gz 2 --train=merge --alpha=0.3
```

What `pautomac_eval.py` does:

- Reads train/test/solution from dir or tar archive.
- Applies historical preprocessing conventions:
  - HMM mode: strips length prefix.
  - PFSA mode: replaces length prefix with dataset alphabet size symbol.
- Trains model, scores with `--output-format=log10 --likelihood=f`.
- Normalizes and computes perplexity vs solution file.
- Writes artifacts under `runs/pautomac/problem_XX/`.

Exact perl normalization compatibility:

```bash
--legacy-normalizer
```

Battery orchestration:

```bash
python3 scripts/pautomac_battery.py --list-models --group all_models_v1
python3 scripts/pautomac_battery.py pautomac_final.tar.gz --group all_models_v1 --problems 1-48 --dry-run
python3 scripts/pautomac_battery.py pautomac_final.tar.gz --group all_models_v1 --problems 1-48 --jobs 4 --keep-going
```

Battery outputs:

- `manifest.json`
- `results.csv`
- `results.jsonl`
- `summary.md`
- per-task logs in `logs/`

CUDA handling in battery script:

- `--cuda-mode auto` skips CUDA-required models if runtime/GPU unavailable.
- `--cuda-mode force` enforces CUDA runs.
- `--cuda-mode off` disables CUDA presets.

## Known Correctness Fixes Already Landed

Phase 1:

- `--t0` parsing fallthrough fixed.
- BW thread count clamped to internal limits.
- Random restart off-by-one fixed.
- HMM parser uninitialized `target` read fixed.
- HMM generation threshold checks fixed.

Phase 2:

- BW log-likelihood accumulation uses shared global mutex.
- HMM normalization corrected to state-wise outgoing transition normalization.

## Remaining Risk / Technical Debt

- CUDA parity/performance against CPU Gibbs is not yet fully validated on modern GPUs.
- Legacy C build still emits compiler warnings; they are known but not yet fully cleaned.
- There is no formal automated test suite yet; validation is currently script/notebook/smoke driven.

## Agent Navigation Playbook

When picking up new tasks, use this order:

1. Read `README.md` + `docs/PYTHON_API_DRAFT.md` for high-level intent.
2. Identify path:
   - C core change: `treba.c`/`gibbs.c`/`dffa.c`.
   - Wrapper UX/API: `treba_py/`.
   - Benchmark/eval flow: `scripts/pautomac_*.py`.
3. Rebuild and smoke-check quickly:

```bash
make clean && make
python3 -m py_compile treba_py/*.py
python3 scripts/pautomac_eval.py --help
python3 scripts/pautomac_battery.py --help
```

4. If touching notebooks, regenerate with jupytext and avoid checkpoint/artifact churn.

## Archive Provenance (Historical Docs)

Primary archive references:

- `https://storage.googleapis.com/google-code-archive/v2/code.google.com/treba/project.json`
- `https://storage.googleapis.com/google-code-archive/v2/code.google.com/treba/downloads-page-1.json`
- `https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/treba/source-archive.zip`

Useful files in archive zip:

- `treba/wiki/TrebaManPage.wiki`
- `treba/wiki/PAutomaC.wiki`

## Repo Hygiene Notes

- Be explicit when staging files; avoid accidentally adding transient notebook outputs/checkpoints.
