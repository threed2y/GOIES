"""
forecaster.py — Crisis forecasting engine
Detects structural signals in the graph and uses LLM to generate potential crisis scenarios.
"""

import json
import re
import requests
import networkx as nx
from typing import Dict, List, Any
from dataclasses import dataclass

OLLAMA_BASE_URL = "http://localhost:11434"
REQUEST_TIMEOUT_SECS = 150

@dataclass
class CrisisForecast:
    rank: int
    title: str
    actors: List[str]
    probability: float
    severity: str
    timeframe: str
    structural_signal: str
    narrative: str
    mitigation: str

@dataclass
class ForecastResult:
    global_risk: float
    global_label: str
    structural_summary: str
    hotspot_nodes: List[str]
    forecasts: List[CrisisForecast]
    model_used: str

def _is_hostile(label: str) -> bool:
    keywords = ["sanction", "attack", "invade", "bomb", "missile", "strike", "kill", "threaten", "blockade", "terrorize", "restrict", "ban", "expel", "dispute", "tension", "pressure", "cyber", "confront"]
    return any(k in label.lower() for k in keywords)

def _is_cooperative(label: str) -> bool:
    keywords = ["cooperate", "ally", "partner", "invest", "aid", "support", "trade", "treaty"]
    return any(k in label.lower() for k in keywords)

def _hotspot_nodes(graph: nx.DiGraph) -> List[str]:
    hostile_counts = {}
    for u, v, d in graph.edges(data=True):
        if _is_hostile(d.get("label", "")):
            hostile_counts[u] = hostile_counts.get(u, 0) + 1
            hostile_counts[v] = hostile_counts.get(v, 0) + 1
    sorted_hotspots = sorted(hostile_counts.items(), key=lambda x: x[1], reverse=True)
    return [node for node, count in sorted_hotspots[:5]]

def _reciprocal_hostility(graph: nx.DiGraph) -> List[str]:
    pairs = []
    edges = list(graph.edges(data=True))
    for i, (u1, v1, d1) in enumerate(edges):
        if not _is_hostile(d1.get("label", "")): continue
        for u2, v2, d2 in edges[i+1:]:
            if u1 == v2 and v1 == u2 and _is_hostile(d2.get("label", "")):
                pairs.append(f"{u1} <-> {v1}")
    return pairs

def _instability_triangles(graph: nx.DiGraph) -> List[str]:
    triangles = []
    nodes = list(graph.nodes)
    # Simple check for triangles of A,B,C where 2 edges hostile and 1 coop
    # Since undirected triangles need graph to be treating directions loosely
    ugraph = graph.to_undirected()
    for n in ugraph.nodes:
        neighbors = list(ugraph.neighbors(n))
        for i in range(len(neighbors)):
            for j in range(i+1, len(neighbors)):
                n1, n2 = neighbors[i], neighbors[j]
                if ugraph.has_edge(n1, n2):
                    # We have a triangle n, n1, n2
                    edges_data = [
                        graph.get_edge_data(n, n1) or graph.get_edge_data(n1, n),
                        graph.get_edge_data(n, n2) or graph.get_edge_data(n2, n),
                        graph.get_edge_data(n1, n2) or graph.get_edge_data(n2, n1)
                    ]
                    
                    hostile_count = 0
                    coop_count = 0
                    for ed in edges_data:
                        if not ed: continue
                        label = list(ed.values())[0].get("label", "") if "label" not in ed else ed.get("label", "")
                        if _is_hostile(label): hostile_count += 1
                        if _is_cooperative(label): coop_count += 1
                        
                    if hostile_count >= 2 and coop_count >= 1:
                        triangles.append(f"{n}-{n1}-{n2}")
    return list(set(triangles))[:5]

def _structural_signals(graph: nx.DiGraph) -> Dict[str, Any]:
    return {
        "hotspots": _hotspot_nodes(graph),
        "reciprocal": _reciprocal_hostility(graph),
        "triangles": _instability_triangles(graph)
    }

def _get_risk_label(score: float) -> str:
    if score >= 75: return "CRITICAL"
    if score >= 50: return "HIGH"
    if score >= 25: return "MEDIUM"
    return "LOW"

def _call_ollama(prompt: str, model: str) -> str:
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=REQUEST_TIMEOUT_SECS,
        )
        response.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Ollama HTTP error: {e}")
    return response.json().get("response", "").strip()

def run_forecast(graph: nx.DiGraph, model: str, focus_query: str = "") -> ForecastResult:
    signals = _structural_signals(graph)
    
    prompt = f"""You are a crisis forecasting intelligence engine. 
Based on the topological structural signals of the current geopolitical graph, generate a crisis forecast.

Focus Area: {focus_query if focus_query else "Global"}

Structural Signals:
- Hotspots: {signals['hotspots']}
- Reciprocal Hostilities: {signals['reciprocal']}
- Instability Triangles: {signals['triangles']}

Output ONLY valid JSON matching this schema:
{{
  "global_risk": 67.0,
  "structural_summary": "1-2 sentence overview of the structural tensions.",
  "forecasts": [
    {{
      "rank": 1,
      "title": "Short Crisis Name",
      "actors": ["Actor1", "Actor2"],
      "probability": 0.75,
      "severity": "CRITICAL",
      "timeframe": "near-term (0-3 months)",
      "structural_signal": "Explain which structural signal led to this.",
      "narrative": "Detailed narrative.",
      "mitigation": "Mitigation step."
    }}
  ]
}}
"""
    raw = _call_ollama(prompt, model)
    
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
    match = re.search(r"\{.*\}", raw.strip(), re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in model output: {raw[:400]}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from model: {e} | Raw: {raw[:400]}")

    global_risk = float(data.get("global_risk", 50.0))
    
    forecasts = []
    for f in data.get("forecasts", []):
        forecasts.append(CrisisForecast(
            rank=int(f.get("rank", 1)),
            title=f.get("title", "Unknown Crisis"),
            actors=f.get("actors", []),
            probability=float(f.get("probability", 0.5)),
            severity=f.get("severity", "MEDIUM"),
            timeframe=f.get("timeframe", "Unknown"),
            structural_signal=f.get("structural_signal", ""),
            narrative=f.get("narrative", ""),
            mitigation=f.get("mitigation", "")
        ))

    return ForecastResult(
        global_risk=global_risk,
        global_label=_get_risk_label(global_risk),
        structural_summary=data.get("structural_summary", ""),
        hotspot_nodes=signals['hotspots'],
        forecasts=forecasts,
        model_used=model
    )
