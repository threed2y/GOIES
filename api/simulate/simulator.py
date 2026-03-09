"""
simulator.py — GOIES Policy Simulation Engine

Given a natural-language policy scenario (e.g. "US lifts sanctions on Iran"),
this module:
  1. Parses the scenario into structured change parameters via LLM
  2. Applies the change to a COPY of the graph (never mutating the live graph)
  3. Runs a second LLM pass to reason about cascading effects
  4. Returns a full SimulationResult with delta nodes/edges and narrative

The live graph is NEVER modified. All changes are hypothetical.
"""
from __future__ import annotations

import json
import re
import requests
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import networkx as nx

from utils import retrieve_graph_context

OLLAMA_BASE_URL = "http://localhost:11434"
REQUEST_TIMEOUT_SECS = 90


# ── Data Models ───────────────────────────────────────────────────────────────
@dataclass
class PolicyChange:
    """A single parsed change to apply to the simulated graph."""

    action: str  # "add_edge", "remove_edge", "add_node", "modify_node"
    source: str = ""
    target: str = ""
    label: str = ""
    node_id: str = ""
    attribute: str = ""
    value: str = ""


@dataclass
class SimulationResult:
    scenario: str
    parsed_changes: List[PolicyChange]
    added_edges: List[Dict[str, Any]]
    removed_edges: List[Dict[str, Any]]
    affected_nodes: List[str]
    cascade_narrative: str
    risk_score: float  # 0–100
    risk_label: str  # LOW / MEDIUM / HIGH / CRITICAL
    second_order: List[str]  # second-order effect predictions
    model_used: str


# ── Risk Labelling ────────────────────────────────────────────────────────────
def _risk_label(score: float) -> str:
    if score >= 75:
        return "CRITICAL"
    if score >= 50:
        return "HIGH"
    if score >= 25:
        return "MEDIUM"
    return "LOW"


# ── LLM Helpers ──────────────────────────────────────────────────────────────
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
        raise TimeoutError("Ollama timed out during simulation.")
    except Exception as e:
        raise RuntimeError(f"Ollama error: {e}")


def _parse_json(raw: str) -> dict:
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in LLM output.")
    return json.loads(m.group(0))


# ── Step 1: Parse Scenario into Graph Changes ─────────────────────────────────
_PARSE_PROMPT = """You are a geopolitical graph mutation engine.

Given a policy scenario, output a JSON object describing graph changes.

ACTIONS:
  add_edge    — new relationship between two entities
  remove_edge — remove existing relationship
  modify_node — change an attribute on an existing node

OUTPUT JSON SCHEMA (return ONLY this JSON, no markdown, no prose):
{{
  "changes": [
    {{"action": "remove_edge", "source": "US", "target": "Iran", "label": "sanctions"}},
    {{"action": "add_edge",    "source": "US", "target": "Iran", "label": "diplomatic talks"}},
    {{"action": "modify_node", "node_id": "Iran", "attribute": "status", "value": "normalising"}}
  ],
  "risk_score": 35,
  "summary": "One-sentence plain-English summary of the policy change"
}}

SCENARIO: {scenario}

EXISTING GRAPH ENTITIES (use these exact names where possible):
{entities}"""


def _parse_scenario(scenario: str, graph: nx.DiGraph, model: str) -> Dict:
    entities = ", ".join(list(graph.nodes)[:40])
    prompt = _PARSE_PROMPT.format(scenario=scenario, entities=entities)
    raw = _call_ollama(prompt, model)
    return _parse_json(raw)


# ── Step 2: Apply Changes to Cloned Graph ────────────────────────────────────
def _apply_changes(
    base_graph: nx.DiGraph,
    changes: List[dict],
) -> tuple[nx.DiGraph, List[PolicyChange], List[Dict], List[Dict]]:
    """Returns (sim_graph, parsed_changes, added_edges, removed_edges)."""
    sim = base_graph.copy()
    parsed: List[PolicyChange] = []
    added: List[Dict] = []
    removed: List[Dict] = []

    for c in changes:
        action = c.get("action", "")

        if action == "add_edge":
            src, tgt, lbl = c.get("source", ""), c.get("target", ""), c.get("label", "")
            if src and tgt:
                if not sim.has_node(src):
                    sim.add_node(src, group="unknown")
                if not sim.has_node(tgt):
                    sim.add_node(tgt, group="unknown")
                sim.add_edge(src, tgt, label=lbl, simulated=True)
                added.append({"from": src, "to": tgt, "label": lbl})
            parsed.append(
                PolicyChange(action="add_edge", source=src, target=tgt, label=lbl)
            )

        elif action == "remove_edge":
            src, tgt = c.get("source", ""), c.get("target", "")
            if sim.has_edge(src, tgt):
                lbl = sim[src][tgt].get("label", "")
                sim.remove_edge(src, tgt)
                removed.append({"from": src, "to": tgt, "label": lbl})
            parsed.append(PolicyChange(action="remove_edge", source=src, target=tgt))

        elif action == "modify_node":
            nid = c.get("node_id", "")
            if sim.has_node(nid):
                sim.nodes[nid][c.get("attribute", "note")] = c.get("value", "")
            parsed.append(
                PolicyChange(
                    action="modify_node",
                    node_id=nid,
                    attribute=c.get("attribute", ""),
                    value=c.get("value", ""),
                )
            )

    return sim, parsed, added, removed


# ── Step 3: LLM Cascade Analysis ─────────────────────────────────────────────
_CASCADE_PROMPT = """You are a senior geopolitical risk analyst.

POLICY SCENARIO: {scenario}

GRAPH CHANGES APPLIED:
{changes_summary}

CURRENT INTELLIGENCE GRAPH CONTEXT:
{graph_context}

Analyse the cascading geopolitical effects of this policy change.

Respond in JSON (no markdown, no prose):
{{
  "cascade_narrative": "3-5 sentence strategic analysis of immediate and medium-term effects",
  "second_order_effects": [
    "Effect 1 on third-party actors",
    "Effect 2 on regional stability",
    "Effect 3 on economic/military balance"
  ],
  "risk_adjustment": 5,
  "key_actors_affected": ["Actor A", "Actor B"]
}}"""


def _cascade_analysis(
    scenario: str,
    changes: List[Dict],
    sim_graph: nx.DiGraph,
    model: str,
    base_risk: float,
) -> Dict:
    changes_summary = "\n".join(
        f"  - {c.get('action','?')}: {c.get('source',c.get('node_id',''))} "
        f"→ {c.get('target','')} [{c.get('label','')}]"
        for c in changes
    )
    ctx = retrieve_graph_context(scenario, sim_graph, max_hops=3, max_edges=25)
    prompt = _CASCADE_PROMPT.format(
        scenario=scenario,
        changes_summary=changes_summary,
        graph_context=ctx,
    )
    raw = _call_ollama(prompt, model)
    try:
        return _parse_json(raw)
    except Exception:
        return {
            "cascade_narrative": raw[:600],
            "second_order_effects": [],
            "risk_adjustment": 0,
            "key_actors_affected": [],
        }


# ── Public API ────────────────────────────────────────────────────────────────
def run_simulation(
    scenario: str,
    graph: nx.DiGraph,
    model: str = "llama3.2",
) -> SimulationResult:
    """
    Main entry point. Runs a full policy simulation against a copy of the graph.
    The live `graph` is NEVER mutated.

    Returns a SimulationResult with narrative, delta edges, risk score.
    """
    if not scenario.strip():
        raise ValueError("Scenario cannot be empty.")

    # Step 1: Parse
    parsed_data = _parse_scenario(scenario, graph, model)
    changes_raw = parsed_data.get("changes", [])
    base_risk = float(parsed_data.get("risk_score", 50))

    # Step 2: Apply to clone
    sim_graph, parsed_changes, added, removed = _apply_changes(graph, changes_raw)

    # Step 3: Cascade analysis
    cascade = _cascade_analysis(scenario, changes_raw, sim_graph, model, base_risk)

    # Final risk score = base ± LLM adjustment, capped 0–100
    final_risk = min(100.0, max(0.0, base_risk + cascade.get("risk_adjustment", 0)))

    # Affected nodes
    affected: List[str] = list(
        {c.get("source", c.get("node_id", "")) or "" for c in changes_raw}
        | {c.get("target", "") or "" for c in changes_raw}
        | set(cascade.get("key_actors_affected", []))
    )
    affected = [a for a in affected if a]

    return SimulationResult(
        scenario=scenario,
        parsed_changes=parsed_changes,
        added_edges=added,
        removed_edges=removed,
        affected_nodes=affected,
        cascade_narrative=cascade.get("cascade_narrative", ""),
        risk_score=final_risk,
        risk_label=_risk_label(final_risk),
        second_order=cascade.get("second_order_effects", []),
        model_used=model,
    )
