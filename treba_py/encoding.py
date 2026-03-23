from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Hashable, Iterable, Sequence

from .config import TokenizationConfig
from .exceptions import AlphabetError
from .types import DatasetInput, TokenT


@dataclass(slots=True)
class EncodedCorpus(Generic[TokenT]):
    sequences: list[list[int]]
    token_to_id: dict[TokenT, int]
    id_to_token: list[TokenT]


class TokenEncoder(Generic[TokenT]):
    """Maps arbitrary hashable symbols to contiguous integer ids expected by Treba."""

    def __init__(self, config: TokenizationConfig | None = None) -> None:
        self.config = config or TokenizationConfig()
        self.token_to_id: dict[TokenT, int] = {}
        self.id_to_token: list[TokenT] = []

    def fit(
        self,
        X: DatasetInput[TokenT],
        *,
        alphabet: Sequence[TokenT] | None = None,
    ) -> TokenEncoder[TokenT]:
        """Fit token mapping from user alphabet or observed symbols in X."""
        tokens: list[TokenT] = []
        if alphabet is not None:
            tokens = list(alphabet)
        else:
            seen: set[TokenT] = set()
            for seq in self._iter_token_sequences(X):
                for token in seq:
                    if token not in seen:
                        seen.add(token)
                        tokens.append(token)

        if not tokens:
            raise AlphabetError("Cannot fit encoder: empty alphabet")

        self.id_to_token = tokens
        self.token_to_id = {token: idx for idx, token in enumerate(tokens)}
        return self

    def transform(self, X: DatasetInput[TokenT]) -> list[list[int]]:
        if not self.token_to_id:
            raise AlphabetError("Encoder is not fitted")

        encoded: list[list[int]] = []
        for seq in self._iter_token_sequences(X):
            row: list[int] = []
            skipped = False
            for token in seq:
                if token in self.token_to_id:
                    row.append(self.token_to_id[token])
                elif self.config.unknown_policy == "ignore_sequence":
                    skipped = True
                    break
                elif self.config.unknown_policy == "use_unk" and self.config.unk_token is not None:
                    unk = self.config.unk_token  # type: ignore[assignment]
                    if unk not in self.token_to_id:
                        raise AlphabetError("unk_token is configured but missing from alphabet")
                    row.append(self.token_to_id[unk])
                else:
                    raise AlphabetError(f"Unknown token: {token!r}")
            if skipped:
                continue
            encoded.append(row)
        return encoded

    def inverse_transform(self, X: Iterable[Sequence[int]]) -> list[list[TokenT]]:
        if not self.id_to_token:
            raise AlphabetError("Encoder is not fitted")
        return [[self.id_to_token[idx] for idx in seq] for seq in X]

    def fit_transform(
        self,
        X: DatasetInput[TokenT],
        *,
        alphabet: Sequence[TokenT] | None = None,
    ) -> EncodedCorpus[TokenT]:
        self.fit(X, alphabet=alphabet)
        encoded = self.transform(X)
        return EncodedCorpus(
            sequences=encoded,
            token_to_id=self.token_to_id.copy(),
            id_to_token=list(self.id_to_token),
        )

    def _iter_token_sequences(self, X: DatasetInput[TokenT]) -> Iterable[Sequence[TokenT]]:
        if isinstance(X, Path):
            if not X.exists():
                raise FileNotFoundError(f"Dataset file not found: {X}")
            with X.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line:
                        yield []
                    else:
                        yield self._split_string_sequence(line)  # type: ignore[misc]
            return

        if isinstance(X, str):
            path = Path(X)
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.rstrip("\n")
                        if not line:
                            yield []
                        else:
                            yield self._split_string_sequence(line)  # type: ignore[misc]
                return
            # Treat a bare string as a single token sequence, not iterable-of-chars.
            yield self._split_string_sequence(X)  # type: ignore[misc]
            return

        for raw in X:  # type: ignore[assignment]
            if isinstance(raw, str):
                yield self._split_string_sequence(raw)  # type: ignore[misc]
            else:
                yield list(raw)

    def _split_string_sequence(self, text: str) -> list[Hashable]:
        if self.config.lowercase:
            text = text.lower()
        if self.config.token_sep == "":
            parts = list(text)
        else:
            parts = text.split(self.config.token_sep)
        if self.config.strip_tokens:
            parts = [part.strip() for part in parts]
        return [part for part in parts if part != ""]
