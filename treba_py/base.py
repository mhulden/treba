from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Generic, Literal, Sequence

from .config import DrawConfig, TokenizationConfig, Topology, TrainingConfig
from .encoding import TokenEncoder
from .exceptions import AlphabetError, NotFittedError
from .model_io import (
    HMMModelData,
    PFSAModelData,
    parse_hmm_model,
    parse_pfsa_model,
)
from .runner import TrebaRunner
from .types import DatasetInput, TokenT
from .viz import draw_hmm, draw_pfsa

LikelihoodMethod = Literal["f", "vit", "b"]
DecodeMethod = Literal["f", "b", "vit"]


@dataclass(slots=True)
class DecodeResult:
    states: list[int]
    prob: float | None = None


@dataclass(slots=True)
class SampleResult(Generic[TokenT]):
    tokens: list[TokenT]
    token_ids: list[int]
    prob: float | None = None
    states: list[int] | None = None


class TrebaModel(Generic[TokenT]):
    """sklearn-style API for Treba-backed models."""

    model_kind: Literal["hmm", "pfsa"]

    def __init__(
        self,
        *,
        n_states: int,
        topology: Topology = "ergodic",
        token_config: TokenizationConfig | None = None,
        training: TrainingConfig | None = None,
        runner: TrebaRunner | None = None,
    ) -> None:
        self.n_states = n_states
        self.topology = topology
        self.token_config = token_config or TokenizationConfig()
        self.training = training or TrainingConfig()
        self.runner = runner or TrebaRunner()

        self.encoder = TokenEncoder[TokenT](self.token_config)
        self.model_text: str | None = None
        self._structured_cache: HMMModelData | PFSAModelData | None = None
        self._model_symbol_ids: set[int] | None = None

    @property
    def is_fitted(self) -> bool:
        return self.model_text is not None

    def fit(
        self,
        X: DatasetInput[TokenT],
        *,
        alphabet: Sequence[TokenT] | None = None,
        sample_weight: Sequence[float] | None = None,
    ) -> TrebaModel[TokenT]:
        """Train this model from sequences.

        Intended behavior:
        - fit token encoder (with optional user alphabet)
        - materialize integer observations file
        - invoke treba training
        - store model text
        """
        encoded = self.encoder.fit_transform(X, alphabet=alphabet).sequences
        if sample_weight is not None:
            encoded = self._apply_sample_weights(encoded, sample_weight)

        with TemporaryDirectory(prefix="treba_py_fit_") as tmpdir:
            tmp = Path(tmpdir)
            obs_path = tmp / "train.obs"
            self._write_observations(obs_path, encoded)
            args = self._build_train_args(alphabet_size=len(self.encoder.id_to_token))
            args.append(str(obs_path))
            result = self.runner.run(args, check=True)
            self.model_text = result.stdout
            self._structured_cache = None
            self._update_model_metadata()
        return self

    def score(
        self,
        X: DatasetInput[TokenT],
        *,
        method: LikelihoodMethod = "f",
        normalized: bool = False,
    ) -> list[float]:
        """Return sequence likelihoods (or log-likelihoods based on training output_format)."""
        self._require_fitted()
        encoded = self.encoder.transform(X)
        self._validate_sequences_against_model(encoded, purpose="score")
        with TemporaryDirectory(prefix="treba_py_score_") as tmpdir:
            tmp = Path(tmpdir)
            model_path = tmp / self._model_filename()
            obs_path = tmp / "score.obs"
            model_path.write_text(self.model_text or "", encoding="utf-8")
            self._write_observations(obs_path, encoded)

            output_format = "log2" if normalized else "real"
            args: list[str] = []
            if self.model_kind == "hmm":
                args.append("--hmm")
            args.extend(
                [
                    f"--output-format={output_format}",
                    f"--likelihood={method}",
                    f"--file={model_path}",
                    str(obs_path),
                ]
            )
            result = self.runner.run(args, check=True)
            values = self._parse_float_lines(result.stdout)
            if not normalized:
                return values
            # Per-token average log-probability in log2-space.
            normed: list[float] = []
            for value, seq in zip(values, encoded, strict=False):
                denom = max(1, len(seq))
                normed.append(value / denom)
            return normed

    def predict_proba(
        self,
        X: DatasetInput[TokenT],
        *,
        method: LikelihoodMethod = "f",
    ) -> list[float]:
        """Alias for `score()` with probability interpretation."""
        return self.score(X, method=method)

    def decode(
        self,
        X: DatasetInput[TokenT],
        *,
        method: DecodeMethod = "vit",
        with_prob: bool = False,
    ) -> list[DecodeResult]:
        """Return best-state paths for each sequence."""
        self._require_fitted()
        encoded = self.encoder.transform(X)
        self._validate_sequences_against_model(encoded, purpose="decode")
        with TemporaryDirectory(prefix="treba_py_decode_") as tmpdir:
            tmp = Path(tmpdir)
            model_path = tmp / self._model_filename()
            obs_path = tmp / "decode.obs"
            model_path.write_text(self.model_text or "", encoding="utf-8")
            self._write_observations(obs_path, encoded)

            decode_arg = f"{method},p" if with_prob else method
            args: list[str] = []
            if self.model_kind == "hmm":
                args.append("--hmm")
            args.extend(
                [
                    f"--decode={decode_arg}",
                    f"--file={model_path}",
                    str(obs_path),
                ]
            )
            result = self.runner.run(args, check=True)
            return self._parse_decode_output(result.stdout, with_prob=with_prob)

    def sample(
        self,
        n: int,
        *,
        with_states: bool = False,
    ) -> list[SampleResult[TokenT]]:
        """Sample sequences from trained model and decode token ids back to symbols."""
        self._require_fitted()
        with TemporaryDirectory(prefix="treba_py_sample_") as tmpdir:
            tmp = Path(tmpdir)
            model_path = tmp / self._model_filename()
            model_path.write_text(self.model_text or "", encoding="utf-8")

            args: list[str] = []
            if self.model_kind == "hmm":
                args.append("--hmm")
            args.extend(
                [
                    f"--generate={n}",
                    f"--file={model_path}",
                ]
            )
            result = self.runner.run(args, check=True)
            return self._parse_sample_output(result.stdout, with_states=with_states)

    def draw(self, *, config: DrawConfig | None = None):
        """Return Graphviz object for inline Jupyter rendering."""
        structured = self.to_structured()
        symbol_map = {i: str(tok) for i, tok in enumerate(self.encoder.id_to_token)}
        if self.model_kind == "hmm":
            return draw_hmm(structured, config=config, symbol_map=symbol_map if symbol_map else None)
        return draw_pfsa(structured, config=config, symbol_map=symbol_map if symbol_map else None)

    def save(self, path: str | Path) -> Path:
        """Persist model text to disk (.hmm or .fsm)."""
        if self.model_text is None:
            raise NotFittedError("Model is not fitted")
        out = Path(path)
        out.write_text(self.model_text, encoding="utf-8")
        return out

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        alphabet: Sequence[TokenT] | None = None,
        token_config: TokenizationConfig | None = None,
        training: TrainingConfig | None = None,
        runner: TrebaRunner | None = None,
    ) -> TrebaModel[TokenT]:
        """Load an already-trained model text file into Python wrapper object.

        `alphabet` lets callers restore token labels for symbol IDs.
        """
        raise NotImplementedError("Use HMM.from_file(...) or PFSA.from_file(...)")

    def to_structured(self) -> HMMModelData | PFSAModelData:
        """Parse raw model text into structured transitions/emissions/finals."""
        self._require_fitted()
        if self._structured_cache is not None:
            return self._structured_cache
        text = self.model_text or ""
        if self.model_kind == "hmm":
            self._structured_cache = parse_hmm_model(text)
        else:
            self._structured_cache = parse_pfsa_model(text)
        return self._structured_cache

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise NotFittedError("Model is not fitted")

    def _model_filename(self) -> str:
        return "model.hmm" if self.model_kind == "hmm" else "model.fsm"

    def _update_model_metadata(self) -> None:
        structured = self.to_structured()
        if isinstance(structured, HMMModelData):
            self._model_symbol_ids = {e.symbol for e in structured.emissions}
        else:
            self._model_symbol_ids = {t.symbol for t in structured.transitions}

    def _validate_sequences_against_model(
        self,
        sequences: Sequence[Sequence[int]],
        *,
        purpose: str,
    ) -> None:
        if self._model_symbol_ids is None:
            self._update_model_metadata()
        allowed = self._model_symbol_ids or set()
        missing: set[int] = set()
        for seq in sequences:
            for symbol_id in seq:
                if symbol_id not in allowed:
                    missing.add(symbol_id)
        if not missing:
            return

        msg = (
            f"Cannot {purpose}: encoded symbols {sorted(missing)} are not in the trained "
            f"model alphabet {sorted(allowed)}."
        )
        if self.token_config.unknown_policy == "use_unk":
            msg += (
                " `unknown_policy='use_unk'` is enabled, but the configured unknown token "
                "was not retained in the trained model alphabet. Ensure unk_token appears in "
                "training data or switch unknown_policy."
            )
        raise AlphabetError(msg)

    def _build_train_args(self, *, alphabet_size: int) -> list[str]:
        cfg = self.training
        args: list[str] = []
        if self.model_kind == "hmm":
            args.append("--hmm")
        args.append(f"--train={cfg.algorithm}")

        if cfg.algorithm not in ("merge", "mdi"):
            initialize = self._build_initialize_arg(alphabet_size)
            args.append(f"--initialize={initialize}")

        if cfg.max_iter is not None:
            args.append(f"--max-iter={cfg.max_iter}")
        if cfg.max_delta is not None:
            args.append(f"--max-delta={cfg.max_delta}")
        if cfg.prior is not None:
            if isinstance(cfg.prior, tuple):
                args.append(f"--prior={cfg.prior[0]},{cfg.prior[1]}")
            else:
                args.append(f"--prior={cfg.prior}")
        if cfg.alpha is not None:
            args.append(f"--alpha={cfg.alpha}")
        if cfg.burnin is not None:
            args.append(f"--burnin={cfg.burnin}")
        if cfg.lag is not None:
            args.append(f"--lag={cfg.lag}")
        if cfg.restarts is not None:
            args.append(f"--restarts={cfg.restarts[0]},{cfg.restarts[1]}")
        if cfg.annealopts is not None:
            args.extend(["-a", f"{cfg.annealopts[0]},{cfg.annealopts[1]},{cfg.annealopts[2]}"])
        if cfg.threads is not None:
            args.append(f"--threads={cfg.threads}")
        if cfg.use_cuda:
            args.append("--cuda")
        if cfg.uniform_probs:
            args.append("--uniform-probs")
        if cfg.output_format:
            args.append(f"--output-format={cfg.output_format}")
        args.extend(cfg.extra_args)
        return args

    def _build_initialize_arg(self, alphabet_size: int) -> str:
        if self.topology == "bakis":
            return f"b{self.n_states},{alphabet_size}"
        if self.topology == "deterministic":
            return f"d{self.n_states},{alphabet_size}"
        return f"{self.n_states},{alphabet_size}"

    @staticmethod
    def _write_observations(path: Path, sequences: Sequence[Sequence[int]]) -> None:
        lines = []
        for seq in sequences:
            lines.append(" ".join(str(x) for x in seq))
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    @staticmethod
    def _parse_float_lines(text: str) -> list[float]:
        vals: list[float] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            vals.append(float(line))
        return vals

    @staticmethod
    def _parse_decode_output(text: str, *, with_prob: bool) -> list[DecodeResult]:
        results: list[DecodeResult] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if with_prob:
                if "\t" in line:
                    prob_s, state_s = line.split("\t", 1)
                else:
                    prob_s, state_s = line.split(maxsplit=1)
                prob = float(prob_s.strip())
                states = [int(x) for x in state_s.strip().split()] if state_s.strip() else []
                results.append(DecodeResult(states=states, prob=prob))
            else:
                states = [int(x) for x in line.split()] if line else []
                results.append(DecodeResult(states=states, prob=None))
        return results

    def _parse_sample_output(self, text: str, *, with_states: bool) -> list[SampleResult[TokenT]]:
        out: list[SampleResult[TokenT]] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            prob = float(parts[0].strip())
            token_ids = [int(x) for x in parts[1].split()] if parts[1].strip() else []
            tokens = [self._decode_token_id(i) for i in token_ids]
            states = None
            if with_states and len(parts) >= 3:
                states = [int(x) for x in parts[2].split()] if parts[2].strip() else []
            out.append(SampleResult(tokens=tokens, token_ids=token_ids, prob=prob, states=states))
        return out

    def _decode_token_id(self, idx: int) -> TokenT:
        if 0 <= idx < len(self.encoder.id_to_token):
            return self.encoder.id_to_token[idx]
        return idx  # type: ignore[return-value]

    @staticmethod
    def _apply_sample_weights(
        sequences: list[list[int]],
        sample_weight: Sequence[float],
    ) -> list[list[int]]:
        if len(sequences) != len(sample_weight):
            raise ValueError("sample_weight length must match number of sequences")
        expanded: list[list[int]] = []
        for seq, w in zip(sequences, sample_weight, strict=True):
            if w <= 0:
                continue
            n = int(round(w))
            if abs(w - n) > 1e-9:
                raise ValueError("sample_weight currently supports only integer-like values")
            expanded.extend([list(seq)] * n)
        return expanded
