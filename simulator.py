"""
simulator.py — Policy simulation engine
Takes a user scenario, applies it to a cloned graph using LLM, and calculates cascade impacts.
"""

import json
import re
import requests
import networkx as nx
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import os

OLLAMA_BASE_URL = "http://localhost:11434"
REQUEST_TIMEOUT_SECS = 120

@dataclass
class SimulationResult:
    scenario: str
    risk_score: float
    risk_label: str
    cascade_narrative: str
    second_order: List[str]
    added_edges: List[Dict[str, str]]
    removed_edges: List[Dict[str, str]]
    affected_nodes: List[str]
    model_used: str

def _call_ollama(prompt: str, model: str) -> str:
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=REQUEST_TIMEOUT_SECS,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(f"Cannot connect to Ollama at {OLLAMA_BASE_URL}.")
    except requests.exceptions.Timeout:
        raise TimeoutError(f"Ollama did not respond within {REQUEST_TIMEOUT_SECS}s.")
    except Exception as e:
        raise RuntimeError(f"Ollama HTTP error: {e}")
    return response.json().get("response", "").strip()

def _extract_json(raw: str) -> Dict[str, Any]:
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
    match = re.search(r"\{.*\}", raw.strip(), re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in model output: {raw[:400]}")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from model: {e} | Raw: {raw[:400]}")

def _parse_scenario(scenario: str, graph_context: str, model: str, persona: str = "strategic policy simulator") -> Dict[str, Any]:
    prompt = f"""You are a {persona}. Parse the given geopolitical scenario into specific graph mutations.

Scenario: "{scenario}"

Current Graph Context:
{graph_context}

Output ONLY valid JSON matching this schema:
{{
  "changes": [
    {{"action": "add_edge", "from": "NodeA", "to": "NodeB", "label": "relationship type"}},
    {{"action": "remove_edge", "from": "NodeA", "to": "NodeB", "label": "relationship type"}},
    {{"action": "modify_node", "node": "NodeA", "attribute": "status", "value": "new status"}}
  ],
  "base_risk": 35  // Integer 0-100 indicating initial systemic risk of this policy
}}
"""
    raw = _call_ollama(prompt, model)
    return _extract_json(raw)

def _apply_changes(graph: nx.DiGraph, changes: List[Dict[str, Any]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[str]]:
    added = []
    removed = []
    affected = set()
    
    for ch in changes:
        action = ch.get("action")
        if action == "add_edge":
            u, v, label = ch.get("from"), ch.get("to"), ch.get("label", "")
            if u and v:
                if not graph.has_node(u): graph.add_node(u, group="unknown")
                if not graph.has_node(v): graph.add_node(v, group="unknown")
                graph.add_edge(u, v, label=label, confidence=0.9)
                added.append({"from": u, "to": v, "label": label})
                affected.update([u, v])
        elif action == "remove_edge":
            u, v = ch.get("from"), ch.get("to")
            if u and v and graph.has_edge(u, v):
                label = graph.edges[u, v].get("label", "unknown")
                graph.remove_edge(u, v)
                removed.append({"from": u, "to": v, "label": label})
                affected.update([u, v])
        elif action == "modify_node":
            node, attr, val = ch.get("node"), ch.get("attribute"), ch.get("value")
            if node and graph.has_node(node):
                if attr and val:
                    graph.nodes[node][attr] = val
                affected.add(node)
                
    return added, removed, list(affected)

def _cascade_analysis(scenario: str, mutations: Dict[str, Any], mutated_graph_info: str, model: str, persona: str = "strategic policy simulator") -> Dict[str, Any]:
    prompt = f"""You are a {persona} evaluating the cascading second-order effects of a policy scenario.

Scenario: "{scenario}"
Mutations applied: {json.dumps(mutations.get('changes', []))}
Mutated Graph Focus:
{mutated_graph_info}

Base Risk was {mutations.get('base_risk', 50)}.

Output ONLY valid JSON matching this schema:
{{
  "cascade_narrative": "A 2-3 sentence strategic overview of the cascading effects.",
  "second_order_effects": ["Effect 1...", "Effect 2...", "Effect 3..."],
  "risk_adjustment": 5, // Integer between -20 and +50 to adjust base risk based on cascades
  "key_actors_affected": ["Actor 1", "Actor 2", "Actor 3"]
}}
"""
    raw = _call_ollama(prompt, model)
    return _extract_json(raw)

def _get_risk_label(score: float) -> str:
    if score >= 75: return "CRITICAL"
    if score >= 50: return "HIGH"
    if score >= 25: return "MEDIUM"
    return "LOW"

def _graph_context_summary(graph: nx.DiGraph, focus_nodes: List[str] | None = None) -> str:
    edges = []
    if focus_nodes:
        for u, v, d in graph.edges(data=True):
            if u in focus_nodes or v in focus_nodes:
                edges.append(f"{u} -> {d.get('label', '')} -> {v}")
    if not edges:
        for u, v, d in list(graph.edges(data=True))[:30]:
            edges.append(f"{u} -> {d.get('label', '')} -> {v}")
    if not edges:
        return "Graph is empty."
    return "\n".join(set(edges[:30]))

def run_simulation(scenario: str, graph: nx.DiGraph, model: str, persona: str = "strategic policy simulator") -> SimulationResult:
    # Pass 1: Parse mutations
    context = _graph_context_summary(graph)
    parsed = _parse_scenario(scenario, context, model, persona)
    
    base_risk = int(parsed.get("base_risk", 50))
    changes = parsed.get("changes", [])
    
    # Apply to a clone
    sim_graph = graph.copy()
    added, removed, affected = _apply_changes(sim_graph, changes)
    
    # Pass 2: Cascade analysis
    mutated_context = _graph_context_summary(sim_graph, focus_nodes=affected)
    cascade = _cascade_analysis(scenario, parsed, mutated_context, model, persona)
    
    risk_adj = int(cascade.get("risk_adjustment", 0))
    final_risk = max(0, min(100, base_risk + risk_adj))
    
    all_affected = list(set(affected + cascade.get("key_actors_affected", [])))
    
    
    result = SimulationResult(
        scenario=scenario,
        risk_score=final_risk,
        risk_label=_get_risk_label(final_risk),
        cascade_narrative=cascade.get("cascade_narrative", ""),
        second_order=cascade.get("second_order_effects", []),
        added_edges=added,
        removed_edges=removed,
        affected_nodes=all_affected,
        model_used=model
    )
    
    # Save to history
    history_file = "sim_history.json"
    history = []
    if os.path.exists(history_file):
        with open(history_file, "r", encoding="utf-8") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                pass
                
    entry = asdict(result)
    entry["timestamp"] = datetime.now().isoformat()
    history.insert(0, entry) # Prepend newest
    
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
        
    return result
