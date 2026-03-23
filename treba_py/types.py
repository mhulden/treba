from __future__ import annotations

from os import PathLike
from typing import Hashable, Iterable, Sequence, TypeVar

TokenT = TypeVar("TokenT", bound=Hashable)

TokenSequence = Sequence[TokenT]
RawSequenceInput = TokenSequence | str
DatasetInput = Iterable[RawSequenceInput] | str | PathLike[str]
