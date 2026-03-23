# Python Wrapper API

`treba_py` provides a notebook-friendly wrapper around the `treba` binary.

## Package Layout

```text
treba_py/
  __init__.py
  base.py
  config.py
  encoding.py
  exceptions.py
  model_io.py
  models.py
  runner.py
  types.py
  viz.py
```

## Main Classes

```python
from treba_py import HMM, PFSA, TrainingConfig, TokenizationConfig, DrawConfig
```

### `HMM` / `PFSA`

```python
class HMM(TrebaModel[TokenT]):
    def __init__(
        *,
        n_states: int,
        topology: Literal["ergodic", "bakis", "deterministic"] = "ergodic",
        token_config: TokenizationConfig | None = None,
        training: TrainingConfig | None = None,
        runner: TrebaRunner | None = None,
    ) -> None: ...

class PFSA(TrebaModel[TokenT]):
    def __init__(
        *,
        n_states: int,
        topology: Literal["ergodic", "bakis", "deterministic"] = "ergodic",
        token_config: TokenizationConfig | None = None,
        training: TrainingConfig | None = None,
        runner: TrebaRunner | None = None,
    ) -> None: ...
```

## Implemented Methods

```python
def fit(
    self,
    X: DatasetInput[TokenT],
    *,
    alphabet: Sequence[TokenT] | None = None,
    sample_weight: Sequence[float] | None = None,
) -> Self: ...

def score(
    self,
    X: DatasetInput[TokenT],
    *,
    method: Literal["f", "vit", "b"] = "f",
    normalized: bool = False,
) -> list[float]: ...

def predict_proba(
    self,
    X: DatasetInput[TokenT],
    *,
    method: Literal["f", "vit", "b"] = "f",
) -> list[float]: ...

def decode(
    self,
    X: DatasetInput[TokenT],
    *,
    method: Literal["vit", "f", "b"] = "vit",
    with_prob: bool = False,
) -> list[DecodeResult]: ...

def sample(self, n: int, *, with_states: bool = False) -> list[SampleResult[TokenT]]: ...

def draw(self, *, config: DrawConfig | None = None): ...

def save(self, path: str | Path) -> Path: ...

@classmethod
def from_file(
    cls,
    path: str | Path,
    *,
    alphabet: Sequence[TokenT] | None = None,
    **kwargs,
) -> Self: ...
```

## Tokenization and Alphabet UX

`TokenEncoder` hides Treba's integer-only input format.

Accepted training/inference data:

- Iterable of token sequences (`list[list[token]]`, tuples, etc.)
- Iterable of strings (split by `token_sep`, default space)
- A text file path (one sequence per line)
- A single string sequence (treated as one sample if path does not exist)

Unknown token handling (`TokenizationConfig.unknown_policy`):

- `error`: raise `AlphabetError`
- `ignore_sequence`: skip sequences containing unknown symbols
- `use_unk`: map unknown symbols to `unk_token`

Important caveat for `use_unk`:

- The `unk_token` symbol id must exist in the trained model alphabet.
- If `unk_token` never appears in training data, Treba may drop that symbol from the learned model; scoring/decoding then raises `AlphabetError` with guidance.

## Visualization

`draw()` returns a `graphviz.Digraph` for notebook display.

- PFSA view: transition labels as `symbol / prob`, optional final-state probabilities
- HMM view: transition probabilities, plus top-k emissions on state labels
- By default, parallel edges are merged (same source/target) and labels are stacked with newlines.
- Defaults are tuned for notebook readability (`graph_size="9,5"`, compact node/edge fonts and node sizes).
- You can override rendering via `DrawConfig`, e.g. `merge_parallel_edges=False` or custom `graph_size`.

## Example

```python
from treba_py import HMM, TrainingConfig

X = [
    "a b c",
    "a b",
    "a c",
]

hmm = HMM(
    n_states=8,
    training=TrainingConfig(algorithm="bw", max_iter=50),
)

hmm.fit(X)
print(hmm.score(["a b c"]))
print(hmm.decode(["a b c"], with_prob=True)[0])

g = hmm.draw()
# In Jupyter, last expression renders graph:
# g
```

## Dependencies

- Core wrapper: Python 3.10+
- Runtime model operations: built `treba` binary available via `./treba` or `PATH` (runner also auto-discovers local `treba` binaries up parent directories)
- Visualization: `pip install graphviz` plus Graphviz system binaries

## Notebook Examples

- `notebooks/01_treba_py_quickstart.ipynb`
- `notebooks/02_pautomac_mini_workflow.ipynb`

Jupytext source files are kept next to the notebooks as `.py` and can be re-rendered with:

```bash
jupytext --to ipynb notebooks/*.py
```
