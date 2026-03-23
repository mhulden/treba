from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class PFSATransition:
    source: int
    target: int
    symbol: int
    prob: float


@dataclass(slots=True)
class PFSAModelData:
    transitions: list[PFSATransition] = field(default_factory=list)
    final_probs: dict[int, float] = field(default_factory=dict)


@dataclass(slots=True)
class HMMTransition:
    source: int
    target: int
    prob: float


@dataclass(slots=True)
class HMMEmission:
    state: int
    symbol: int
    prob: float


@dataclass(slots=True)
class HMMModelData:
    transitions: list[HMMTransition] = field(default_factory=list)
    emissions: list[HMMEmission] = field(default_factory=list)


def parse_pfsa_model(text: str) -> PFSAModelData:
    """Parse Treba/OpenFST-like PFSA text model into structured data."""
    model = PFSAModelData()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 4:
            source, target, symbol = int(parts[0]), int(parts[1]), int(parts[2])
            prob = float(parts[3])
            model.transitions.append(PFSATransition(source, target, symbol, prob))
        elif len(parts) == 3:
            source, target, symbol = int(parts[0]), int(parts[1]), int(parts[2])
            model.transitions.append(PFSATransition(source, target, symbol, 1.0))
        elif len(parts) == 2:
            state, prob = int(parts[0]), float(parts[1])
            model.final_probs[state] = prob
        elif len(parts) == 1:
            model.final_probs[int(parts[0])] = 1.0
        else:
            raise ValueError(f"Unrecognized PFSA model line: {line!r}")
    return model


def parse_hmm_model(text: str) -> HMMModelData:
    """Parse Treba HMM text model into structured data."""
    model = HMMModelData()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 3 and parts[1] == ">":
            source = int(parts[0])
            target = int(parts[2])
            prob = float(parts[3]) if len(parts) >= 4 else 1.0
            model.transitions.append(HMMTransition(source, target, prob))
        else:
            if len(parts) == 3:
                state, symbol, prob = int(parts[0]), int(parts[1]), float(parts[2])
            elif len(parts) == 2:
                state, symbol, prob = int(parts[0]), int(parts[1]), 1.0
            else:
                raise ValueError(f"Unrecognized HMM model line: {line!r}")
            model.emissions.append(HMMEmission(state, symbol, prob))
    return model


def serialize_pfsa_model(model: PFSAModelData) -> str:
    lines: list[str] = []
    for t in sorted(model.transitions, key=lambda x: (x.source, x.target, x.symbol)):
        lines.append(f"{t.source} {t.target} {t.symbol} {t.prob:.17g}")
    for state in sorted(model.final_probs):
        lines.append(f"{state} {model.final_probs[state]:.17g}")
    return "\n".join(lines) + ("\n" if lines else "")


def serialize_hmm_model(model: HMMModelData) -> str:
    lines: list[str] = []
    for t in sorted(model.transitions, key=lambda x: (x.source, x.target)):
        lines.append(f"{t.source} > {t.target} {t.prob:.17g}")
    for e in sorted(model.emissions, key=lambda x: (x.state, x.symbol)):
        lines.append(f"{e.state} {e.symbol} {e.prob:.17g}")
    return "\n".join(lines) + ("\n" if lines else "")


def write_model(path: str | Path, text: str) -> Path:
    p = Path(path)
    p.write_text(text, encoding="utf-8")
    return p


def read_model(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")
