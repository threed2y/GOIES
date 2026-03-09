"""
forecaster.py — GOIES Crisis Forecasting Engine

Combines structural graph analysis with LLM reasoning to produce
a ranked list of crisis forecasts with probability estimates.

Structural signals used:
  - High betweenness nodes (single points of failure / escalation hubs)
  - Dense hostile-edge clusters (hotspot detection)
  - Isolated nodes with sudden high-degree growth (surprise actors)
  - Reciprocal hostile edges (direct confrontation pairs)
  - Triangles with two hostile + one cooperative edge (instability triangle)
  - Country nodes under incoming hostile pressure
"""

from __future__ import annotations

import json
import re
import requests
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import networkx as nx

from utils import retrieve_graph_context

OLLAMA_BASE_URL = "http://localhost:11434"
REQUEST_TIMEOUT_SECS = 90

HOSTILE_RE = re.compile(
    r"sanction|attack|invad|bomb|missile|strike|kill|threaten|"
    r"wage war|blockade|seize|destabil|terroris|restrict|ban|"
    r"expel|confront|dispute|tension|pressure|isolat|cyber",
    re.I,
)


# ── Data Models ───────────────────────────────────────────────────────────────
@dataclass
class CrisisForecast:
    rank: int
    title: str
    actors: List[str]
    probability: float  # 0–1
    severity: str  # LOW / MEDIUM / HIGH / CRITICAL
    timeframe: str  # e.g. "near-term (0–3 months)"
    structural_signal: str  # what graph pattern triggered this
    narrative: str
    mitigation: str


@dataclass
class ForecastReport:
    forecasts: List[CrisisForecast]
    global_risk: float  # weighted aggregate 0–100
    global_label: str
    hotspot_nodes: List[str]
    structural_summary: str
    model_used: str


# ── Structural Analysis ───────────────────────────────────────────────────────
def _hostile_edges(graph: nx.DiGraph) -> List[Tuple[str, str, str]]:
    return [
        (u, v, d.get("label", ""))
        for u, v, d in graph.edges(data=True)
        if HOSTILE_RE.search(d.get("label", ""))
    ]


def _reciprocal_hostility(graph: nx.DiGraph) -> List[Tuple[str, str]]:
    """Find node pairs with hostile edges in both directions."""
    hostile = {(u, v) for u, v, _ in _hostile_edges(graph)}
    return [(u, v) for (u, v) in hostile if (v, u) in hostile and u < v]


def _hotspot_nodes(graph: nx.DiGraph, top_n: int = 8) -> List[str]:
    """Nodes with highest hostile edge count."""
    counts: Dict[str, int] = {}
    for u, v, _ in _hostile_edges(graph):
        counts[u] = counts.get(u, 0) + 2  # aggressors weighted higher
        counts[v] = counts.get(v, 0) + 1
    return sorted(counts, key=lambda x: -counts[x])[:top_n]


def _instability_triangles(graph: nx.DiGraph) -> List[Tuple[str, str, str]]:
    """Triangles where 2 edges are hostile and 1 is cooperative — classic instability."""
    cooperative_re = re.compile(r"cooperat|ally|partner|trade|invest|support", re.I)
    triangles: List[Tuple[str, str, str]] = []
    ug = graph.to_undirected()
    for a, b, c in set(
        tuple(sorted(tri))
        for tri in [
            sorted([u, v, w])
            for u in ug
            for v in ug.neighbors(u)
            for w in ug.neighbors(v)
            if w in ug.neighbors(u) and u != w
        ]
    ):
        edges = [(a, b), (b, c), (a, c), (b, a), (c, b), (c, a)]
        labels = [
            graph[u][v].get("label", "") if graph.has_edge(u, v) else ""
            for u, v in edges
        ]
        hostile_count = sum(1 for l in labels if HOSTILE_RE.search(l))
        coop_count = sum(1 for l in labels if cooperative_re.search(l))
        if hostile_count >= 2 and coop_count >= 1:
            triangles.append((a, b, c))
    return triangles[:5]


def _structural_signals(graph: nx.DiGraph) -> Dict[str, Any]:
    """Run all structural detectors, return summary dict."""
    n = len(graph.nodes)
    if n < 3:
        return {"hotspots": [], "reciprocal": [], "triangles": [], "betweenness": []}

    hotspots = _hotspot_nodes(graph)
    reciprocal = _reciprocal_hostility(graph)
    triangles = _instability_triangles(graph)

    betweenness: List[Tuple[str, float]] = []
    if n >= 5:
        try:
            bc = nx.betweenness_centrality(graph)
            # Only return nodes that are also in hostile edges
            hostile_nodes = {u for u, v, _ in _hostile_edges(graph)} | {
                v for u, v, _ in _hostile_edges(graph)
            }
            betweenness = sorted(
                [(k, v) for k, v in bc.items() if k in hostile_nodes],
                key=lambda x: -x[1],
            )[:5]
        except Exception:
            pass

    return {
        "hotspots": hotspots,
        "reciprocal": reciprocal,
        "triangles": triangles,
        "betweenness": betweenness,
        "hostile_edge_count": len(_hostile_edges(graph)),
    }


# ── LLM Forecast Generation ───────────────────────────────────────────────────
_FORECAST_PROMPT = """You are a senior geopolitical crisis analyst with 30 years of experience.

STRUCTURAL INTELLIGENCE SIGNALS from the knowledge graph:
- Hotspot nodes (most hostile relationships): {hotspots}
- Direct confrontation pairs (bilateral hostility): {reciprocal}
- Instability triangles (2 hostile + 1 cooperative relationship): {triangles}
- High-betweenness conflict brokers: {betweenness}
- Total hostile edge count: {hostile_count}

GRAPH CONTEXT:
{graph_context}

Based on these structural signals and graph context, forecast the TOP 4 most likely geopolitical crises.

Respond ONLY with this JSON (no markdown, no prose):
{{
  "global_risk": 62,
  "structural_summary": "2–3 sentence assessment of the overall threat landscape",
  "forecasts": [
    {{
      "rank": 1,
      "title": "Short crisis title",
      "actors": ["Actor A", "Actor B"],
      "probability": 0.72,
      "severity": "HIGH",
      "timeframe": "near-term (0–3 months)",
      "structural_signal": "Reciprocal hostility + high betweenness of Actor A",
      "narrative": "2–3 sentence explanation of why this crisis is likely",
      "mitigation": "1 sentence recommended diplomatic or policy action"
    }}
  ]
}}

SEVERITY OPTIONS: LOW, MEDIUM, HIGH, CRITICAL
TIMEFRAME OPTIONS: immediate (0–4 weeks), near-term (0–3 months), medium-term (3–12 months), long-term (1–3 years)"""


def _severity_label(prob: float) -> str:
    if prob >= 0.75:
        return "CRITICAL"
    if prob >= 0.55:
        return "HIGH"
    if prob >= 0.35:
        return "MEDIUM"
    return "LOW"


def _call_ollama(prompt: str, model: str) -> str:
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=REQUEST_TIMEOUT_SECS,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise ConnectionError("Ollama not reachable.")
    except requests.exceptions.Timeout:
        raise TimeoutError("Ollama timed out during forecasting.")


def _parse_json(raw: str) -> dict:
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise ValueError("No JSON in forecast output.")
    return json.loads(m.group(0))


# ── Public API ────────────────────────────────────────────────────────────────
def run_forecast(
    graph: nx.DiGraph,
    model: str = "llama3.2",
    focus_query: str = "",
) -> ForecastReport:
    """
    Runs the full crisis forecasting pipeline.
    Returns a ForecastReport with ranked CrisisForecasts.
    """
    if len(graph.nodes) < 3:
        return ForecastReport(
            forecasts=[],
            global_risk=0.0,
            global_label="LOW",
            hotspot_nodes=[],
            structural_summary="Insufficient graph data for forecasting.",
            model_used=model,
        )

    signals = _structural_signals(graph)
    ctx = retrieve_graph_context(
        focus_query or "crisis conflict tension war sanctions",
        graph,
        max_hops=3,
        max_edges=30,
    )

    prompt = _FORECAST_PROMPT.format(
        hotspots=", ".join(signals["hotspots"][:6]) or "none detected",
        reciprocal="; ".join(f"{a}↔{b}" for a, b in signals["reciprocal"][:4])
        or "none detected",
        triangles="; ".join(f"{a}–{b}–{c}" for a, b, c in signals["triangles"][:3])
        or "none detected",
        betweenness=", ".join(f"{n}({s:.2f})" for n, s in signals["betweenness"][:4])
        or "none",
        hostile_count=signals["hostile_edge_count"],
        graph_context=ctx,
    )

    raw = _call_ollama(prompt, model)

    try:
        data = _parse_json(raw)
    except Exception:
        return ForecastReport(
            forecasts=[],
            global_risk=50.0,
            global_label="MEDIUM",
            hotspot_nodes=signals["hotspots"],
            structural_summary="LLM output could not be parsed. Check Ollama status.",
            model_used=model,
        )

    forecasts: List[CrisisForecast] = []
    for f in data.get("forecasts", []):
        prob = float(f.get("probability", 0.5))
        forecasts.append(
            CrisisForecast(
                rank=int(f.get("rank", len(forecasts) + 1)),
                title=f.get("title", "Unnamed Crisis"),
                actors=f.get("actors", []),
                probability=prob,
                severity=f.get("severity", _severity_label(prob)),
                timeframe=f.get("timeframe", "near-term"),
                structural_signal=f.get("structural_signal", ""),
                narrative=f.get("narrative", ""),
                mitigation=f.get("mitigation", ""),
            )
        )

    global_risk = float(data.get("global_risk", 50))
    global_label = (
        "CRITICAL"
        if global_risk >= 75
        else "HIGH"
        if global_risk >= 50
        else "MEDIUM"
        if global_risk >= 25
        else "LOW"
    )

    return ForecastReport(
        forecasts=forecasts,
        global_risk=global_risk,
        global_label=global_label,
        hotspot_nodes=signals["hotspots"],
        structural_summary=data.get("structural_summary", ""),
        model_used=model,
    )
