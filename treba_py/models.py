from __future__ import annotations

from pathlib import Path
from typing import Generic, Sequence

from .base import TrebaModel
from .config import TokenizationConfig, Topology, TrainingConfig
from .exceptions import AlphabetError
from .model_io import parse_hmm_model, parse_pfsa_model
from .runner import TrebaRunner
from .types import TokenT


def _set_encoder_for_loaded_model(
    model: TrebaModel[TokenT],
    *,
    symbol_ids: set[int],
    alphabet: Sequence[TokenT] | None,
) -> None:
    if not symbol_ids:
        return
    max_symbol = max(symbol_ids)

    if alphabet is None:
        # Identity mapping for integer symbol IDs in serialized model.
        id_to_token = list(range(max_symbol + 1))  # type: ignore[assignment]
    else:
        id_to_token = list(alphabet)
        if len(id_to_token) <= max_symbol:
            raise AlphabetError(
                f"Provided alphabet has {len(id_to_token)} entries, "
                f"but model references symbol id {max_symbol}."
            )

    model.encoder.id_to_token = id_to_token
    model.encoder.token_to_id = {tok: idx for idx, tok in enumerate(id_to_token)}


class HMM(TrebaModel[TokenT], Generic[TokenT]):
    model_kind = "hmm"

    def __init__(
        self,
        *,
        n_states: int,
        topology: Topology = "ergodic",
        token_config: TokenizationConfig | None = None,
        training: TrainingConfig | None = None,
        runner: TrebaRunner | None = None,
    ) -> None:
        super().__init__(
            n_states=n_states,
            topology=topology,
            token_config=token_config,
            training=training,
            runner=runner,
        )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        alphabet: Sequence[TokenT] | None = None,
        token_config: TokenizationConfig | None = None,
        training: TrainingConfig | None = None,
        runner: TrebaRunner | None = None,
    ) -> HMM[TokenT]:
        text = Path(path).read_text(encoding="utf-8")
        parsed = parse_hmm_model(text)
        states: set[int] = set()
        symbols: set[int] = set()
        for t in parsed.transitions:
            states.add(t.source)
            states.add(t.target)
        for e in parsed.emissions:
            states.add(e.state)
            symbols.add(e.symbol)
        n_states = max(states) + 1 if states else 1

        model = cls(
            n_states=n_states,
            token_config=token_config,
            training=training,
            runner=runner,
        )
        model.model_text = text
        model._structured_cache = parsed
        model._update_model_metadata()
        _set_encoder_for_loaded_model(model, symbol_ids=symbols, alphabet=alphabet)
        return model


class PFSA(TrebaModel[TokenT], Generic[TokenT]):
    model_kind = "pfsa"

    def __init__(
        self,
        *,
        n_states: int,
        topology: Topology = "ergodic",
        token_config: TokenizationConfig | None = None,
        training: TrainingConfig | None = None,
        runner: TrebaRunner | None = None,
    ) -> None:
        super().__init__(
            n_states=n_states,
            topology=topology,
            token_config=token_config,
            training=training,
            runner=runner,
        )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        alphabet: Sequence[TokenT] | None = None,
        token_config: TokenizationConfig | None = None,
        training: TrainingConfig | None = None,
        runner: TrebaRunner | None = None,
    ) -> PFSA[TokenT]:
        text = Path(path).read_text(encoding="utf-8")
        parsed = parse_pfsa_model(text)
        states: set[int] = set(parsed.final_probs.keys())
        symbols: set[int] = set()
        for t in parsed.transitions:
            states.add(t.source)
            states.add(t.target)
            symbols.add(t.symbol)
        n_states = max(states) + 1 if states else 1

        model = cls(
            n_states=n_states,
            token_config=token_config,
            training=training,
            runner=runner,
        )
        model.model_text = text
        model._structured_cache = parsed
        model._update_model_metadata()
        _set_encoder_for_loaded_model(model, symbol_ids=symbols, alphabet=alphabet)
        return model
