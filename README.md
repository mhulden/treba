# Treba - PFSA/HMM Training and Decoding

<p align="center">
  <img src="docs/treba-logo.png" alt="Treba logo" width="248">
</p>

`treba` is a command-line toolkit (C, optional CUDA) for probabilistic finite-state automata (PFSA/WFSA) and hidden Markov models (HMM).

This README is reconstructed from the original Google Code archive documentation and adapted for this repository.

## What Treba Does

Treba supports four core tasks:

1. Training (`--train=...`)
2. Likelihood calculation (`--likelihood=...`)
3. Decoding (`--decode=...`)
4. Sequence generation (`--generate=NUM`)

Implemented training algorithms include:

- Baum-Welch / EM (`bw`)
- Deterministic annealing Baum-Welch (`dabw`)
- Viterbi training / hard EM (`vit`, `vitbw`)
- Variational Bayes-style updates (`vb`)
- Collapsed Gibbs sampling (`gs`), with optional CUDA acceleration
- Deterministic PFSA induction by state merging (`merge`, `mdi`)

## Build

From source:

```bash
make
sudo make install
```

With CUDA support (legacy build path):

```bash
make CUDA=1
sudo make install
```

When switching between CPU and CUDA builds, use `make clean` first to avoid stale objects.

Notes for modern systems:

- Treba needs GSL (`-lgsl -lgslcblas`) and headers.
- This repository Makefile has been updated for modern toolchains:
  - GSL detection:
    - Linux/Unix: tries `pkg-config gsl` first, then falls back to `-lgsl -lgslcblas`
    - macOS: auto-detects Homebrew prefix (`/opt/homebrew` or `/usr/local`) and adds include/lib flags
  - CUDA arch defaults: `sm_60 sm_70 sm_75 sm_80 sm_86`
  - CUDA libs searched in `$(CUDA_INSTALL_PATH)/lib64` and `.../lib`
  - Legacy GCC global-symbol behavior preserved via `-fcommon`
- On `aarch64`, build uses an NVCC compatibility workaround (`-D__GNUC__=8 -D__GNUC_MINOR__=0`) for CUDA 12 + recent glibc headers.

### macOS (Homebrew) quick notes

Install GSL:

```bash
brew install gsl
```

Then build normally:

```bash
make clean && make
```

If your GSL prefix is nonstandard, override explicitly:

```bash
make GSL_PREFIX="$(brew --prefix)"
```

You can also override flags directly:

```bash
make GSL_CFLAGS="-I/your/prefix/include" GSL_LIBS="-L/your/prefix/lib -lgsl -lgslcblas"
```

## CLI Overview

Training:

```bash
treba --train=bw --initialize=10 observations.obs > model.fsm
```

HMM training:

```bash
treba --hmm --train=bw --initialize=25 observations.obs > model.hmm
```

Likelihood:

```bash
treba --likelihood=f --file=model.fsm observations.obs
treba --hmm --likelihood=f --file=model.hmm observations.obs
```

Decoding:

```bash
treba --decode=vit --file=model.fsm observations.obs
treba --hmm --decode=vit,p --file=model.hmm observations.obs
```

Generation:

```bash
treba --generate=100 --file=model.fsm
treba --hmm --generate=100 --file=model.hmm
```

## Data Formats

### Observations

One sequence per line, whitespace-separated integer symbols. Empty line = empty string.

Example:

```text
1 7 4 3
3 4 3 2

5 5 5 1 0
```

### PFSA files

Transition line:

```text
SOURCE TARGET SYMBOL PROB
```

Final-probability line:

```text
STATE PROB
```

### HMM files

Transition line:

```text
SOURCE > TARGET TRANSITION_PROB
```

Emission line:

```text
STATE SYMBOL EMISSION_PROB
```

Conventions used by Treba docs/code:

- State `0` is start.
- Highest-numbered state is end.
- Emissions are from non-start/non-end states.

## PAutomaC 2012 Workflow

The original Treba docs include a PAutomaC-oriented workflow (48 problems; each with train/test/solution files).

Original downloadable bundle: `treba-pautomac.tar.gz` (archive downloads), which contains:

- `pautomac-data/` (problems 1..48)
- `pautomac.perl` wrapper

Wrapper usage:

```bash
./pautomac.perl PAUTOMAC_PATH PROBLEM_NUMBER TREBA_ARGUMENTS...
```

Example:

```bash
./pautomac.perl ./pautomac-data 2 --train=merge --alpha=0.3
```

Important wrapper behavior:

- For HMM runs (`--hmm`), it strips the PAutomaC line-length prefix.
- For PFSA runs, it replaces the line-length prefix with the dataset alphabet size symbol.
- It evaluates predictions against the provided solution file via perplexity.

### Python Replacement Wrapper (this repo)

The historical `pautomac.perl` flow is now available as Python:

```bash
python3 scripts/pautomac_eval.py --treba-bin ./treba pautomac_final.tar.gz 2 --train=merge --alpha=0.3
```

This command:

- reads train/test/solution for the selected problem (from directory or tar archive),
- applies the same PFSA/HMM preprocessing convention as the perl wrapper,
- trains Treba with your provided args,
- scores test strings with `--output-format=log10 --likelihood=f`,
- normalizes probabilities and reports perplexity.

Artifacts are written under `runs/pautomac/problem_XX/`.

To reproduce the perl wrapper normalization logic exactly, add `--legacy-normalizer`.

### Battery Runner (all models x selected problems)

A configurable battery runner is included for repeatable benchmark sweeps:

```bash
python3 scripts/pautomac_battery.py --list-models --group all_models_v1
```

Plan a full sweep without executing:

```bash
python3 scripts/pautomac_battery.py pautomac_final.tar.gz \
  --group all_models_v1 \
  --problems 1-48 \
  --dry-run
```

Run a full sweep (example with 4 workers):

```bash
python3 scripts/pautomac_battery.py pautomac_final.tar.gz \
  --group all_models_v1 \
  --problems 1-48 \
  --jobs 4 \
  --keep-going
```

Notes:

- Presets live in `scripts/pautomac_battery_presets.json`.
- CUDA preset models are auto-skipped when CUDA runtime is not usable (`--cuda-mode auto`, default).
- Each run writes `manifest.json`, `results.csv`, `results.jsonl`, `summary.md`, and per-task logs under `runs/pautomac_battery/<run_id>/`.

## Python API

A notebook-friendly Python wrapper (classes `HMM`/`PFSA`, token encoding, runner backend, and Graphviz hooks) is available in:

- `treba_py/`
- `docs/PYTHON_API_DRAFT.md`

Current capabilities:

- Train with `fit(...)` using arbitrary hashable Python tokens
- Score with `score(...)` / `predict_proba(...)`
- Decode with `decode(...)`
- Sample with `sample(...)`
- Render models with Graphviz via `draw(...)`
- Save/load with `save(...)` and `HMM.from_file(...)` / `PFSA.from_file(...)`

Install the wrapper for notebook use:

```bash
python3 -m pip install -e .
```

Notes:

- Unknown-token mapping (`unknown_policy='use_unk'`) requires the unknown symbol to be represented in the trained model alphabet; see `docs/PYTHON_API_DRAFT.md` for details.

Notebook examples:

- `notebooks/01_treba_py_quickstart.ipynb` (toy PFSA/HMM train/score/decode/draw/save/load)
- `notebooks/02_pautomac_mini_workflow.ipynb` (small PAutomaC workflow on one problem)
- Source-of-truth jupytext files:
  - `notebooks/01_treba_py_quickstart.py`
  - `notebooks/02_pautomac_mini_workflow.py`

Regenerate notebooks from jupytext sources:

```bash
jupytext --to ipynb notebooks/*.py
```

## Legacy Release Notes (from archive metadata)

- `treba-1.01.tar.gz` (2013-11-28 UTC): source + binaries
- `treba-pautomac.tar.gz` (2013-12-12 UTC): PAutomaC data + wrapper
- `treba-0.1.tar.gz` (2012-07-13 UTC): early source + binaries

## Documentation Pointers

- Local man page: `man/treba.1`
- Archived wiki copy in source archive:
  - `wiki/TrebaManPage.wiki`
  - `wiki/PAutomaC.wiki`

## License

GPLv2 (see `COPYING`).
