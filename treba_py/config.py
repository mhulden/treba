from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Literal

TrainAlgorithm = Literal["merge", "mdi", "bw", "dabw", "gs", "vb", "vit", "vitbw"]
Topology = Literal["ergodic", "bakis", "deterministic"]
UnknownPolicy = Literal["error", "ignore_sequence", "use_unk"]


@dataclass(slots=True)
class TokenizationConfig:
    token_sep: str = " "
    lowercase: bool = False
    strip_tokens: bool = True
    unk_token: Hashable | None = None
    unknown_policy: UnknownPolicy = "error"


@dataclass(slots=True)
class TrainingConfig:
    algorithm: TrainAlgorithm = "bw"
    max_iter: int = 200
    max_delta: float | None = None
    prior: float | tuple[float, float] | None = None
    alpha: float | None = None
    burnin: int | None = None
    lag: int | None = None
    restarts: tuple[int, int] | None = None
    annealopts: tuple[float, float, float] | None = None
    threads: int | str | None = None
    use_cuda: bool = False
    uniform_probs: bool = False
    output_format: Literal["real", "log10", "ln", "log2", "nlog10", "nln", "nlog2"] = "real"
    extra_args: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DrawConfig:
    engine: str = "dot"
    rankdir: Literal["LR", "TB"] = "LR"
    min_prob: float = 0.0
    max_edges: int | None = None
    top_k_emissions: int = 5
    show_final_probs: bool = True
    prob_precision: int = 4
    # Merge parallel edges (same source and target) and stack labels with newlines.
    merge_parallel_edges: bool = True
    edge_label_sep: str = "\\n"
    # Graph sizing defaults tuned for notebook display.
    graph_size: str | None = "9,5"
    graph_ratio: str | None = "compress"
    graph_margin: float = 0.03
    node_fontsize: int = 10
    edge_fontsize: int = 9
    node_width: float = 0.35
    node_height: float = 0.35
