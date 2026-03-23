from __future__ import annotations

from collections import defaultdict
from typing import Any

from .config import DrawConfig
from .model_io import HMMModelData, PFSAModelData


def draw_pfsa(
    model: PFSAModelData,
    *,
    config: DrawConfig | None = None,
    symbol_map: dict[int, str] | None = None,
) -> Any:
    """Return a graphviz graph for a PFSA model.

    Returns a `graphviz.Digraph`.
    """
    cfg = config or DrawConfig()
    try:
        from graphviz import Digraph
    except ImportError as exc:
        raise ImportError("Graphviz Python package is required for draw(); install `graphviz`.") from exc

    g = Digraph("PFSA", engine=cfg.engine)
    graph_attrs: dict[str, str] = {"rankdir": cfg.rankdir}
    if cfg.graph_size is not None:
        graph_attrs["size"] = cfg.graph_size
    if cfg.graph_ratio is not None:
        graph_attrs["ratio"] = cfg.graph_ratio
    graph_attrs["margin"] = str(cfg.graph_margin)
    graph_attrs["pad"] = "0.02"
    g.attr("graph", **graph_attrs)
    g.attr(
        "node",
        fontsize=str(cfg.node_fontsize),
        width=f"{cfg.node_width:.3g}",
        height=f"{cfg.node_height:.3g}",
    )
    g.attr("edge", fontsize=str(cfg.edge_fontsize))
    g.node("__start__", shape="point", label="")

    states = set(model.final_probs.keys())
    for t in model.transitions:
        states.add(t.source)
        states.add(t.target)
    if 0 in states:
        g.edge("__start__", "0")

    for s in sorted(states):
        if cfg.show_final_probs and s in model.final_probs:
            label = f"{s}\\nF={model.final_probs[s]:.{cfg.prob_precision}g}"
            g.node(str(s), label=label, shape="doublecircle")
        else:
            g.node(str(s), label=str(s), shape="circle")

    trans = [t for t in model.transitions if t.prob >= cfg.min_prob]
    if cfg.merge_parallel_edges:
        grouped: dict[tuple[int, int], list[tuple[int, float]]] = defaultdict(list)
        for t in trans:
            grouped[(t.source, t.target)].append((t.symbol, t.prob))

        grouped_items = list(grouped.items())
        grouped_items.sort(
            key=lambda item: max(prob for _, prob in item[1]),
            reverse=True,
        )
        if cfg.max_edges is not None:
            grouped_items = grouped_items[: cfg.max_edges]

        for (src, dst), lines in grouped_items:
            lines.sort(key=lambda x: x[1], reverse=True)
            parts: list[str] = []
            for symbol_id, prob in lines:
                sym = symbol_map[symbol_id] if symbol_map and symbol_id in symbol_map else str(symbol_id)
                parts.append(f"{sym} / {prob:.{cfg.prob_precision}g}")
            g.edge(str(src), str(dst), label=cfg.edge_label_sep.join(parts))
    else:
        trans.sort(key=lambda x: x.prob, reverse=True)
        if cfg.max_edges is not None:
            trans = trans[: cfg.max_edges]

        for t in trans:
            sym = symbol_map[t.symbol] if symbol_map and t.symbol in symbol_map else str(t.symbol)
            label = f"{sym} / {t.prob:.{cfg.prob_precision}g}"
            g.edge(str(t.source), str(t.target), label=label)
    return g


def draw_hmm(
    model: HMMModelData,
    *,
    config: DrawConfig | None = None,
    symbol_map: dict[int, str] | None = None,
) -> Any:
    """Return a graphviz graph for an HMM model.

    Returns a `graphviz.Digraph`.
    """
    cfg = config or DrawConfig()
    try:
        from graphviz import Digraph
    except ImportError as exc:
        raise ImportError("Graphviz Python package is required for draw(); install `graphviz`.") from exc

    g = Digraph("HMM", engine=cfg.engine)
    graph_attrs: dict[str, str] = {"rankdir": cfg.rankdir}
    if cfg.graph_size is not None:
        graph_attrs["size"] = cfg.graph_size
    if cfg.graph_ratio is not None:
        graph_attrs["ratio"] = cfg.graph_ratio
    graph_attrs["margin"] = str(cfg.graph_margin)
    graph_attrs["pad"] = "0.02"
    g.attr("graph", **graph_attrs)
    g.attr(
        "node",
        fontsize=str(cfg.node_fontsize),
        width=f"{cfg.node_width:.3g}",
        height=f"{cfg.node_height:.3g}",
    )
    g.attr("edge", fontsize=str(cfg.edge_fontsize))

    states: set[int] = set()
    for t in model.transitions:
        states.add(t.source)
        states.add(t.target)
    for e in model.emissions:
        states.add(e.state)

    emissions_by_state: dict[int, list[tuple[int, float]]] = {}
    for e in model.emissions:
        emissions_by_state.setdefault(e.state, []).append((e.symbol, e.prob))

    for s in sorted(states):
        label = str(s)
        if s in emissions_by_state and cfg.top_k_emissions > 0:
            items = sorted(emissions_by_state[s], key=lambda x: x[1], reverse=True)[: cfg.top_k_emissions]
            emissions_text: list[str] = []
            for sym, prob in items:
                sym_text = symbol_map[sym] if symbol_map and sym in symbol_map else str(sym)
                emissions_text.append(f"{sym_text}:{prob:.{cfg.prob_precision}g}")
            if emissions_text:
                label = f"{s}\\n" + "\\n".join(emissions_text)
        g.node(str(s), label=label, shape="circle")

    trans = [t for t in model.transitions if t.prob >= cfg.min_prob]
    if cfg.merge_parallel_edges:
        grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
        for t in trans:
            grouped[(t.source, t.target)].append(t.prob)

        grouped_items = list(grouped.items())
        grouped_items.sort(key=lambda item: max(item[1]), reverse=True)
        if cfg.max_edges is not None:
            grouped_items = grouped_items[: cfg.max_edges]

        for (src, dst), probs in grouped_items:
            probs.sort(reverse=True)
            label = cfg.edge_label_sep.join(f"{p:.{cfg.prob_precision}g}" for p in probs)
            g.edge(str(src), str(dst), label=label)
    else:
        trans.sort(key=lambda x: x.prob, reverse=True)
        if cfg.max_edges is not None:
            trans = trans[: cfg.max_edges]
        for t in trans:
            g.edge(str(t.source), str(t.target), label=f"{t.prob:.{cfg.prob_precision}g}")

    return g
