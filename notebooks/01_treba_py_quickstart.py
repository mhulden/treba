# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Treba Python Wrapper Quickstart
#
# This notebook shows a few small, notebook-friendly workflows with `treba_py`:
#
# 1. Train and inspect a PFSA from tokenized symbol sequences.
# 2. Train an HMM from character-level strings.
# 3. Save and reload models with `from_file(...)`.

# %%
from pathlib import Path
import sys

REPO_ROOT = Path.cwd()
if not (REPO_ROOT / "treba").exists() and (REPO_ROOT.parent / "treba").exists():
    REPO_ROOT = REPO_ROOT.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from treba_py import DrawConfig, HMM, PFSA, TokenizationConfig, TrainingConfig, TrebaRunner

TREBA_BIN = (REPO_ROOT / "treba").resolve()
print("Repo root:", REPO_ROOT)
print("Using Treba binary:", TREBA_BIN)
RUNNER = TrebaRunner(str(TREBA_BIN))

# %% [markdown]
# ## 1) PFSA on toy token sequences

# %%
X_pfsa = [
    ["walk", "shop", "clean"],
    ["walk", "shop"],
    ["walk", "clean"],
    ["clean", "shop"],
    ["walk", "walk", "shop"],
]

pfsa = PFSA(
    n_states=6,
    training=TrainingConfig(algorithm="bw", max_iter=30),
    runner=RUNNER,
)

pfsa.fit(X_pfsa)
print("PFSA fitted:", pfsa.is_fitted)

# %%
print("Likelihoods:", pfsa.score([["walk", "shop"], ["clean", "shop"]]))
print("Normalized log2 scores:", pfsa.score([["walk", "shop"]], normalized=True))

# %%
decoded = pfsa.decode([["walk", "shop", "clean"]], with_prob=True)
print(decoded[0])

# %%
samples = pfsa.sample(3, with_states=True)
samples

# %%
pfsa.draw(config=DrawConfig(rankdir="LR", top_k_emissions=3, prob_precision=3))

# %% [markdown]
# ## 2) HMM on character-level strings
#
# Setting `token_sep=""` means each character is treated as a token.

# %%
X_hmm = [
    "abba",
    "aba",
    "baba",
    "abba",
    "baab",
]

hmm = HMM(
    n_states=5,
    token_config=TokenizationConfig(token_sep=""),
    training=TrainingConfig(algorithm="bw", max_iter=25),
    runner=RUNNER,
)

hmm.fit(X_hmm)
print("HMM fitted:", hmm.is_fitted)

# %%
print("HMM score:", hmm.score(["abba", "baba"]))
print("HMM decode:", hmm.decode(["abba"], with_prob=True)[0])

# %%
hmm.draw(config=DrawConfig(rankdir="LR", top_k_emissions=4, prob_precision=3))

# %% [markdown]
# ## 3) Save and reload

# %%
artifacts = REPO_ROOT / "notebooks" / "artifacts"
artifacts.mkdir(parents=True, exist_ok=True)

pfsa_path = artifacts / "toy_pfsa.fsm"
hmm_path = artifacts / "toy_hmm.hmm"

pfsa.save(pfsa_path)
hmm.save(hmm_path)

print("Saved:", pfsa_path, hmm_path)

# %%
# Pass an alphabet to restore human-readable labels on reload.
pfsa2 = PFSA.from_file(pfsa_path, alphabet=["walk", "shop", "clean"])
print("Reloaded PFSA score:", pfsa2.score([["walk", "shop"]]))

# For HMM char-level toy data, provide alphabet in symbol-id order.
hmm2 = HMM.from_file(
    hmm_path,
    alphabet=["a", "b"],
    token_config=TokenizationConfig(token_sep=""),
)
print("Reloaded HMM score:", hmm2.score(["abba"]))

# %% [markdown]
# ## Notes
#
# - For unknown symbols, configure `TokenizationConfig(unknown_policy=...)`.
# - If using `unknown_policy="use_unk"`, ensure the unknown token appears in training data,
#   otherwise Treba may drop that symbol from the trained alphabet.
# - For larger benchmark sweeps, use the `scripts/pautomac_eval.py` and
#   `scripts/pautomac_battery.py` tooling.
