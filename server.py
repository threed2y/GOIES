"""
server.py — GOIES FastAPI Backend v3
New in this version:
  - /api/geo         — country positions + tension metrics
  - /api/simulate    — policy simulation (graph clone, LLM cascade)
  - /api/forecast    — crisis forecasting (structural + LLM)
"""

from __future__ import annotations
import io, pathlib, re
from typing import Any, Dict, List, Optional
import networkx as nx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from extractor import check_ollama_health, extract_intelligence, list_available_models
from utils import (
    export_csv,
    export_graphml,
    export_json,
    get_ego_subgraph,
    get_graph_analytics,
    load_graph,
    resolve_node_name,
    retrieve_graph_context,
    save_graph,
)
from geo import get_geo_data
from simulator import run_simulation
from forecaster import run_forecast

app = FastAPI(title="GOIES", version="3.0.0", docs_url="/api/docs")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

graph: nx.DiGraph = load_graph()
MAX_INPUT_CHARS = 8_000

GROUP_COLORS: Dict[str, str] = {
    "country": "#ff7b72",
    "person": "#ffa657",
    "organization": "#d2a8ff",
    "technology": "#79c0ff",
    "event": "#7ee787",
    "treaty": "#f0e68c",
    "resource": "#56d364",
    "unknown": "#8b949e",
}


def _fmt_tooltip(group, attributes, confidence):
    color = GROUP_COLORS.get(group, "#8b949e")
    lines = [f'<b style="color:{color};font-family:monospace">{group.upper()}</b>']
    for k, v in attributes.items():
        lines.append(f'<span style="color:#64748b">{k}:</span> {v}')
    lines.append(f'<span style="color:#64748b">confidence:</span> {confidence:.2f}')
    return "<br>".join(lines)


def graph_to_vis(g: nx.DiGraph):
    nodes = []
    for node_id, data in g.nodes(data=True):
        group = data.get("group", "unknown")
        color = GROUP_COLORS.get(group, "#8b949e")
        conf = data.get("confidence", 1.0)
        try:
            attrs = eval(data.get("title", "{}"))
        except:
            attrs = {}
        nodes.append(
            {
                "id": node_id,
                "label": node_id,
                "group": group,
                "color": {
                    "background": color,
                    "border": color,
                    "highlight": {"background": "#ffffff", "border": color},
                    "hover": {"background": color, "border": "#ffffff"},
                },
                "title": _fmt_tooltip(group, attrs, conf),
                "confidence": conf,
                "size": 16,
                "borderWidth": 2,
                "font": {"color": "#e2e8f0", "size": 13},
                "shadow": {
                    "enabled": True,
                    "color": color + "44",
                    "size": 12,
                    "x": 0,
                    "y": 0,
                },
            }
        )
    edges = []
    for u, v, data in g.edges(data=True):
        edges.append(
            {
                "from": u,
                "to": v,
                "label": data.get("label", ""),
                "arrows": "to",
                "color": {
                    "color": "#1e3a5f",
                    "highlight": "#00e5ff",
                    "hover": "#00e5ff",
                    "inherit": False,
                },
                "font": {
                    "color": "#3d5a7a",
                    "size": 10,
                    "align": "middle",
                    "strokeWidth": 0,
                },
                "width": 1.5,
                "smooth": {"type": "continuous"},
                "confidence": data.get("confidence", 1.0),
            }
        )
    return {"nodes": nodes, "edges": edges}


def _update_graph(extractions):
    nodes_added, edges_added, new_ids = 0, 0, []
    for ext in extractions:
        cls = ext.extraction_class.lower()
        if cls in {
            "country",
            "person",
            "organization",
            "technology",
            "event",
            "treaty",
            "resource",
        }:
            canonical = resolve_node_name(graph, ext.extraction_text)
            if not graph.has_node(canonical):
                nodes_added += 1
                new_ids.append(canonical)
            graph.add_node(
                canonical,
                title=str(ext.attributes),
                group=cls,
                confidence=ext.confidence,
            )
        elif cls == "relationship":
            src_raw = ext.attributes.get("source", "")
            tgt_raw = ext.attributes.get("target", "")
            if not src_raw or not tgt_raw:
                continue
            src = resolve_node_name(graph, src_raw)
            tgt = resolve_node_name(graph, tgt_raw)
            for n in (src, tgt):
                if not graph.has_node(n):
                    graph.add_node(n, group="unknown")
            if not graph.has_edge(src, tgt):
                edges_added += 1
            graph.add_edge(
                src, tgt, label=ext.extraction_text, confidence=ext.confidence
            )
    save_graph(graph)
    return {
        "nodes_added": nodes_added,
        "edges_added": edges_added,
        "new_node_ids": new_ids,
    }


# ── Request Models ─────────────────────────────────────────────────────────────
class ExtractRequest(BaseModel):
    text: str
    model: str = "llama3.2"


class QueryRequest(BaseModel):
    question: str
    model: str = "llama3.2"


class SimulateRequest(BaseModel):
    scenario: str
    model: str = "llama3.2"


class ForecastRequest(BaseModel):
    model: str = "llama3.2"
    focus: str = ""


# ── Existing Endpoints ─────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return check_ollama_health()


@app.get("/api/models")
def models():
    return {"models": list_available_models()}


@app.get("/api/graph")
def get_graph_ep(ego: Optional[str] = None, hops: int = 2):
    g = get_ego_subgraph(graph, ego, hops) if ego and ego in graph else graph
    return {
        "vis": graph_to_vis(g),
        "analytics": get_graph_analytics(graph),
        "filtered": ego is not None and ego in graph,
    }


@app.post("/api/extract")
def extract(req: ExtractRequest):
    if not req.text.strip():
        raise HTTPException(400, "Text cannot be empty.")
    if len(req.text) > MAX_INPUT_CHARS:
        raise HTTPException(400, f"Input exceeds {MAX_INPUT_CHARS:,} chars.")
    try:
        extractions = extract_intelligence(req.text, model=req.model)
    except ConnectionError as e:
        raise HTTPException(503, str(e))
    except TimeoutError as e:
        raise HTTPException(504, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))
    diff = _update_graph(extractions)
    entities = sum(
        1 for e in extractions if e.extraction_class.lower() != "relationship"
    )
    return {
        "extractions": len(extractions),
        "entities": entities,
        "relations": len(extractions) - entities,
        **diff,
        "vis": graph_to_vis(graph),
        "analytics": get_graph_analytics(graph),
    }


@app.post("/api/query")
def query(req: QueryRequest):
    import requests as http

    if len(graph.nodes) == 0:
        return {"answer": "Graph is empty. Ingest data first.", "context": ""}
    context = retrieve_graph_context(req.question, graph)
    prompt = (
        "You are a senior geopolitical intelligence analyst. "
        "Answer using ONLY the Knowledge Graph Context. "
        'If insufficient, say "Insufficient data in current intelligence graph."\n\n'
        f"Knowledge Graph Context:\n{context}\n\nQuestion: {req.question}\n\nConcise strategic answer:"
    )
    try:
        resp = http.post(
            "http://localhost:11434/api/generate",
            json={"model": req.model, "prompt": prompt, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        answer = resp.json().get("response", "No response.")
    except Exception as e:
        raise HTTPException(503, f"Ollama error: {e}")
    return {"answer": answer, "context": context}


@app.delete("/api/graph")
def clear_graph():
    graph.clear()
    save_graph(graph)
    return {"status": "cleared"}


@app.get("/api/export/{fmt}")
def export(fmt: str):
    if fmt == "json":
        return StreamingResponse(
            io.StringIO(export_json(graph)),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=goies_graph.json"},
        )
    elif fmt == "csv":
        return StreamingResponse(
            io.StringIO(export_csv(graph)),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=goies_edges.csv"},
        )
    elif fmt == "graphml":
        return StreamingResponse(
            io.BytesIO(export_graphml(graph)),
            media_type="application/xml",
            headers={"Content-Disposition": "attachment; filename=goies_graph.graphml"},
        )
    raise HTTPException(400, f"Unknown format: {fmt}")


# ── NEW: Geo Endpoint ──────────────────────────────────────────────────────────
@app.get("/api/geo")
def get_geo():
    markers = get_geo_data(graph)
    return {"markers": markers, "total": len(markers)}


# ── NEW: Policy Simulation ─────────────────────────────────────────────────────
@app.post("/api/simulate")
def simulate(req: SimulateRequest):
    if not req.scenario.strip():
        raise HTTPException(400, "Scenario cannot be empty.")
    if len(graph.nodes) == 0:
        raise HTTPException(400, "Graph is empty. Ingest data first.")
    try:
        result = run_simulation(req.scenario, graph, model=req.model)
    except ConnectionError as e:
        raise HTTPException(503, str(e))
    except TimeoutError as e:
        raise HTTPException(504, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))
    return {
        "scenario": result.scenario,
        "risk_score": result.risk_score,
        "risk_label": result.risk_label,
        "cascade_narrative": result.cascade_narrative,
        "second_order": result.second_order,
        "added_edges": result.added_edges,
        "removed_edges": result.removed_edges,
        "affected_nodes": result.affected_nodes,
        "model_used": result.model_used,
    }


# ── NEW: Crisis Forecast ───────────────────────────────────────────────────────
@app.post("/api/forecast")
def forecast(req: ForecastRequest):
    if len(graph.nodes) < 3:
        raise HTTPException(400, "Need at least 3 nodes to generate a forecast.")
    try:
        result = run_forecast(graph, model=req.model, focus_query=req.focus)
    except ConnectionError as e:
        raise HTTPException(503, str(e))
    except TimeoutError as e:
        raise HTTPException(504, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))
    return {
        "global_risk": result.global_risk,
        "global_label": result.global_label,
        "structural_summary": result.structural_summary,
        "hotspot_nodes": result.hotspot_nodes,
        "model_used": result.model_used,
        "forecasts": [
            {
                "rank": f.rank,
                "title": f.title,
                "actors": f.actors,
                "probability": f.probability,
                "severity": f.severity,
                "timeframe": f.timeframe,
                "structural_signal": f.structural_signal,
                "narrative": f.narrative,
                "mitigation": f.mitigation,
            }
            for f in result.forecasts
        ],
    }


# ── Static SPA ─────────────────────────────────────────────────────────────────
_static = pathlib.Path(__file__).parent / "static"
_static.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(_static), html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
