# ◈ GOIES — Global Ontology Intelligence Engine
### Production Architecture & Feature Roadmap

> **Status:** Active Development · v3 (Browser SPA) deployed at [tanu-1403.github.io/GOIES](https://tanu-1403.github.io/GOIES)  
> **Stack:** FastAPI · vis.js · Leaflet.js · Ollama (llama3.2) · NetworkX · Python 3.11+

---

## Table of Contents

1. [Vision](#1-vision)
2. [Production Architecture](#2-production-architecture)
3. [Core Intelligence Pipeline](#3-core-intelligence-pipeline)
4. [Knowledge Graph Engine](#4-knowledge-graph-engine)
5. [Geo-Positional System](#5-geo-positional-system)
6. [Policy Simulation Engine](#6-policy-simulation-engine)
7. [Crisis Forecasting Engine](#7-crisis-forecasting-engine)
8. [Strategic Analyst Chat](#8-strategic-analyst-chat)
9. [Frontend — Command Interface](#9-frontend--command-interface)
10. [Data Persistence & Storage](#10-data-persistence--storage)
11. [Multi-Model & LLM Layer](#11-multi-model--llm-layer)
12. [Security & Access Control](#12-security--access-control)
13. [Observability & Monitoring](#13-observability--monitoring)
14. [Testing Strategy](#14-testing-strategy)
15. [Deployment](#15-deployment)
16. [Feature Backlog](#16-feature-backlog)
17. [File Structure](#17-file-structure)
18. [API Reference](#18-api-reference)

---

## 1. Vision

GOIES is an **open-source geopolitical intelligence platform** that transforms raw text — news articles, diplomatic cables, OSINT reports — into an interactive, queryable knowledge graph. It enables analysts to:

- **Map** entity networks (countries, people, organizations, treaties, events) from unstructured text
- **Visualize** geopolitical tension as a live world map with tension heatmaps
- **Simulate** policy scenarios and model cascading geopolitical effects
- **Forecast** crises using structural graph analysis combined with LLM reasoning
- **Query** the graph in natural language via a GraphRAG analyst interface

The system is designed to run **entirely locally** — no data ever leaves the user's machine. The LLM runs via Ollama. The graph is persisted locally. Privacy is a first-class constraint.

---

## 2. Production Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GOIES Production                          │
├─────────────────┬───────────────────────┬───────────────────────┤
│   Frontend SPA  │    FastAPI Backend     │   Intelligence Layer  │
│                 │                       │                       │
│  vis.js graph   │  /api/extract         │  extractor.py         │
│  Leaflet map    │  /api/query           │  geo.py               │
│  Simulation UI  │  /api/simulate        │  simulator.py         │
│  Forecast UI    │  /api/forecast        │  forecaster.py        │
│  Analyst chat   │  /api/geo             │  utils.py             │
│  Timeline view  │  /api/timeline        │  timeline.py          │
│  Path finder    │  /api/path            │  graph_algo.py        │
│  Report gen     │  /api/report          │  reporter.py          │
└────────┬────────┴──────────┬────────────┴──────────┬────────────┘
         │                   │                        │
         │                   ▼                        ▼
         │         ┌─────────────────┐    ┌──────────────────────┐
         │         │  NetworkX Graph  │    │   Ollama REST API    │
         │         │  (in-memory +   │    │   localhost:11434    │
         │         │   JSON persist) │    │   llama3.2 / gemma2  │
         │         └─────────────────┘    │   mistral / etc.     │
         │                                └──────────────────────┘
         ▼
  ┌─────────────────┐
  │  Static Hosting  │
  │  GitHub Pages    │
  │  (fallback mode) │
  └─────────────────┘
```

### Two Deployment Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Static (GitHub Pages)** | Single `index.html`, calls Ollama directly from browser | Personal use, demos, zero-infra |
| **Full Stack** | FastAPI + `uvicorn`, serves SPA, full Python backend | Teams, production, heavier workloads |

---

## 3. Core Intelligence Pipeline

### 3.1 Entity Extraction

**Current:** JSON-mode Ollama prompt → 7 entity classes + relationships

**Production upgrades:**

#### Chunking Strategy
```python
# utils.py
def chunk_text(text: str, max_chars=4000, overlap=200) -> List[str]:
    """Sentence-boundary aware chunking with overlap for context continuity."""
```
- Chunks respect sentence boundaries (no mid-sentence splits)
- 200-character overlap ensures cross-chunk entity linking
- Chunk progress displayed in UI with `[CHUNK 1/3]` badges

#### Entity Classes
```
Country          — nation-states, territories, regions
Person           — politicians, military leaders, diplomats
Organization     — governments, militaries, NGOs, corporations, blocs
Technology       — weapons systems, surveillance tech, cyber tools
Event            — wars, elections, summits, crises, attacks
Treaty           — agreements, sanctions regimes, alliances
Resource         — oil, minerals, financial flows, data
```

#### Confidence Filtering
- Each extraction carries a confidence score `0.0–1.0`
- Items below `0.5` are silently dropped
- Scores `0.5–0.75` shown with a muted visual indicator
- Scores `>0.75` rendered at full opacity

#### Fuzzy Entity Resolution
```python
# utils.py — prevents "US", "United States", "USA" becoming 3 separate nodes
def resolve_node_name(graph, raw_name, threshold=0.82) -> str:
    """SequenceMatcher-based fuzzy resolution. Returns canonical name."""
```

### 3.2 Ingestion Sources (Roadmap)

| Source | Status | Implementation |
|--------|--------|----------------|
| Plain text paste | ✅ Done | textarea input |
| URL scraping | 🔲 Planned | `httpx` + `readability-lxml` |
| RSS/Atom feeds | 🔲 Planned | `feedparser` → auto-poll |
| PDF upload | 🔲 Planned | `pypdf2` text extraction |
| DOCX upload | 🔲 Planned | `python-docx` |
| Bulk folder import | 🔲 Planned | directory watcher |
| Twitter/X feed | 🔲 Planned | Nitter RSS proxy |

### 3.3 Streaming Extraction (Roadmap)
```
POST /api/extract/stream  →  text/event-stream
data: {"chunk": 1, "entity": "Russia", "type": "country", "confidence": 0.94}
data: {"chunk": 1, "relationship": "threatens", "from": "Russia", "to": "Ukraine"}
data: {"done": true, "totals": {"entities": 12, "relationships": 8}}
```
- Real-time token streaming to extraction log via SSE
- UI shows entities popping into graph as they're extracted, not all at once

---

## 4. Knowledge Graph Engine

### 4.1 Graph Data Model

```python
# Node schema
{
  "id": "Russia",
  "group": "country",         # entity class
  "confidence": 0.94,
  "attributes": {
    "capital": "Moscow",
    "leader": "Putin"
  },
  "ingested_at": "2026-03-09T11:00:00Z",   # temporal layer
  "source_count": 3,                         # how many articles mentioned this
  "tension_score": 78.4                      # computed by geo.py
}

# Edge schema
{
  "from": "Russia",
  "to": "Ukraine",
  "label": "invades",
  "confidence": 0.97,
  "ingested_at": "2026-03-09T11:00:00Z",
  "source": "Reuters, 2026-03-01"
}
```

### 4.2 Graph Analytics

Computed on every write via `utils.get_graph_analytics()`:

| Metric | Algorithm | Use |
|--------|-----------|-----|
| Degree centrality | NetworkX `degree_centrality` | Most connected actors |
| Betweenness centrality | NetworkX `betweenness_centrality` | Bridge/broker nodes |
| Weakly connected components | Union-Find | Isolated clusters |
| Graph density | `2E / N(N-1)` | Network cohesion |
| Top degree nodes | Sorted degree dict | Quick actor ranking |

### 4.3 Ego Subgraph (Implemented)
```
Double-click any node → BFS 2-hop subgraph filter
Esc → return to full graph
GET /api/graph?ego=Russia&hops=2
```

### 4.4 Path Finder (Roadmap)
```
GET /api/path?from=US&to=Iran&algorithm=shortest
→ Returns: US → sanctions → Iran
   Or:     US → Israel → funds → IDF → threatens → Iran
```
- UI: pick two nodes from dropdowns or click on graph
- Shortest path highlighted as glowing animated trail in vis.js
- All paths up to length 4 enumerable
- Paths can be exported as a narrative: "US is connected to Iran via 3 hops through Israel"

### 4.5 Graph Versioning (Roadmap)

```
goies_graph_v1_2026-03-01.json
goies_graph_v2_2026-03-05.json
goies_graph_v3_2026-03-09.json
```

- Every extraction creates a timestamped snapshot
- Timeline slider in HUD to scrub between versions
- Diff view: nodes/edges added/removed between two snapshots highlighted in green/red
- "Play" mode animates graph evolution over time

### 4.6 Entity Deduplication (Roadmap)

```
Node Merge UI:
  Drag "United States" onto "US"
  → Confirm merge dialog
  → All edges reattributed to canonical node
  → Source node deleted
  → Operation logged in graph history
```

### 4.7 Conflict Detection (Roadmap)

```python
# Detect contradictory edges
def detect_conflicts(graph) -> List[Conflict]:
    """
    Flags edge pairs where the same pair of nodes have
    contradictory relationship types.
    Example: Russia → [supports] → Syria AND Russia → [bombs] → Syria
    """
```
- Contradictions displayed as orange dashed edges in vis.js
- Alert badge in HUD when conflicts are detected
- Analyst can tag conflicts as "resolved" or "escalated"

---

## 5. Geo-Positional System

### 5.1 Tension Metric Algorithm

```python
def calculate_country_tensions(graph) -> Dict[str, float]:
    for node in country_nodes:
        score = 0
        
        # Outgoing hostile edges (aggressor weight)
        for edge in outgoing_edges(node):
            score += edge_score(edge.label) * 1.2
        
        # Incoming hostile pressure (target weight)  
        for edge in incoming_edges(node):
            score += edge_score(edge.label) * 0.9
        
        # Military event bonus
        for event_node in connected_events(node):
            if is_military(event_node):
                score += 7.0
        
        # Degree centrality multiplier
        score *= (1.0 + 0.04 * min(degree(node), 20))
    
    return normalize_to_100(scores)
```

#### Edge Score Vocabulary

| Score | Keywords |
|-------|----------|
| +18 | sanction, attack, invade, bomb, missile, strike, kill, threaten, blockade, terrorize |
| +9 | restrict, ban, expel, dispute, tension, pressure, cyber, confront |
| -3 | cooperate, ally, partner, invest, aid, support, trade, treaty |
| +2 | (default / unknown) |

### 5.2 Tension Color Bands

| Score | Color | Label |
|-------|-------|-------|
| 75–100 | `#ff2244` | Critical |
| 50–75 | `#ff6b35` | High |
| 25–50 | `#ffaa40` | Medium |
| 10–25 | `#ffe066` | Low |
| 0–10 | `#00ff88` | Peaceful |

### 5.3 Map Features (Current)
- CartoDB Dark Matter tiles (Leaflet.js)
- Circle markers sized `6px + tension × 0.2` (max 26px)
- Fill opacity `0.3 + tension/100 × 0.5`
- Dashed glow ring at tension > 50
- Popup cards: entity name, type, tension score, connection count, top 5 edges
- Country name labels with tension-colored text shadow

### 5.4 Map Features (Roadmap)

#### Tension Heatmap Layer
- Choropleth coloring of country polygons (GeoJSON borders)
- Toggle between circle markers and choropleth fill
- Country borders drawn proportional to tension level

#### Tension Over Time
- Scrub timeline slider → map updates to show historical tension state
- Animated playback of tension evolution
- Alert when a country crosses a tension threshold

#### Tension Alerts
```python
# config
TENSION_ALERTS = {
    "Iran": {"threshold": 70, "notify": True},
    "Taiwan": {"threshold": 60, "notify": True},
}
```
- Browser notification when watched entity crosses threshold
- HUD pulsing badge for any entity at CRITICAL

---

## 6. Policy Simulation Engine

### 6.1 How It Works

```
User input: "The US lifts all sanctions on Iran"
     ↓
Pass 1 (LLM): Parse scenario into structured mutations
{
  "changes": [
    {"action": "remove_edge", "from": "US", "to": "Iran", "label": "sanctions"},
    {"action": "add_edge", "from": "US", "to": "Iran", "label": "diplomatic talks"},
    {"action": "modify_node", "node": "Iran", "attribute": "status", "value": "normalising"}
  ],
  "base_risk": 35
}
     ↓
Apply to GRAPH COPY (live graph never mutated)
     ↓
Pass 2 (LLM): Cascade analysis using modified graph context
{
  "cascade_narrative": "...",
  "second_order_effects": ["...", "...", "..."],
  "risk_adjustment": +5,
  "key_actors_affected": ["Israel", "Saudi Arabia", "IAEA"]
}
     ↓
Output: SimulationResult with risk score, delta, narrative
```

### 6.2 Risk Scoring

```
Final Risk = base_risk + llm_risk_adjustment
           = clamped to [0, 100]
           = labeled LOW / MEDIUM / HIGH / CRITICAL
```

| Label | Range | Visual |
|-------|-------|--------|
| LOW | 0–24 | Green `#00ff99` |
| MEDIUM | 25–49 | Amber `#ffb347` |
| HIGH | 50–74 | Orange `#ff6b35` |
| CRITICAL | 75–100 | Red `#ff3355` pulsing |

### 6.3 Planned Enhancements

#### Multi-Step Simulation (Roadmap)
```
Step 1: "US lifts Iran sanctions"
Step 2: "Israel responds with military exercises"
Step 3: "Iran accelerates nuclear enrichment"
→ Compound risk timeline showing escalation chain
```

#### Simulation History (Roadmap)
- All simulations saved with timestamp + scenario text
- Compare two simulations side-by-side
- Export simulation as a PDF policy brief

#### Monte Carlo Mode (Roadmap)
- Run same scenario 10× with temperature=1.0
- Show distribution of risk scores
- Probability distribution of second-order outcomes

---

## 7. Crisis Forecasting Engine

### 7.1 Structural Signal Detection

```python
# forecaster.py — runs BEFORE the LLM call
signals = {
    "hotspots":    _hotspot_nodes(graph),        # highest hostile edge count
    "reciprocal":  _reciprocal_hostility(graph), # A→B and B→A both hostile
    "triangles":   _instability_triangles(graph), # 2 hostile + 1 coop = risk
    "betweenness": _conflict_brokers(graph),      # high betweenness in hostile cluster
    "hostile_count": len(_hostile_edges(graph))
}
```

#### Instability Triangle Logic
A triangle of 3 actors where:
- 2 edges are hostile (sanctions, attacks, threatens)  
- 1 edge is cooperative (ally, partner, trade)

This is a classic balance-theory instability pattern — the cooperative relationship is under pressure from both hostile sides.

### 7.2 Forecast Output Schema

```python
@dataclass
class CrisisForecast:
    rank:              int
    title:             str           # "Taiwan Strait Military Escalation"
    actors:            List[str]     # ["China", "Taiwan", "US"]
    probability:       float         # 0.72
    severity:          str           # "HIGH"
    timeframe:         str           # "near-term (0-3 months)"
    structural_signal: str           # "Reciprocal hostility + high betweenness of China"
    narrative:         str           # 2-3 sentence explanation
    mitigation:        str           # "Diplomatic back-channels through Singapore..."
```

### 7.3 Planned Enhancements

#### Historical Calibration (Roadmap)
- Compare model forecasts against known historical outcomes
- Brier score tracking per model
- Confidence interval display: "72% ± 8%"

#### Scenario-Conditional Forecasting (Roadmap)
```
"If US imposes Taiwan sanctions, what crises become more/less likely?"
→ Run forecast on simulated graph copy
→ Show delta: crises that increased, decreased, or newly appeared
```

#### Forecast Subscriptions (Roadmap)
- Watch list: "Alert me if a Taiwan forecast exceeds 60% probability"
- Weekly digest: top 5 forecasts summarized as a PDF brief

---

## 8. Strategic Analyst Chat

### 8.1 GraphRAG Architecture

```
User query: "How are Russia and NATO connected?"
     ↓
BFS context retrieval:
  - Find nodes matching query keywords
  - Expand 2 hops from matched nodes  
  - Collect up to 25 edges as context
     ↓
LLM prompt:
  "You are a geopolitical analyst.
   Graph context: [BFS edges]
   Question: [user query]
   Concise strategic answer:"
     ↓
Response streamed to chat window
```

### 8.2 Current Features
- Multi-turn chat with full history
- Per-message "view graph context" toggle
- Typing indicator (`···`)
- Enter-to-send

### 8.3 Planned Enhancements

#### Conversation Memory (Roadmap)
- Chat history persisted to localStorage / disk
- "Continue analysis from last session"
- Named sessions: "Ukraine Crisis Analysis", "Iran Nuclear Track"

#### Entity Pinning (Roadmap)
- Pin entities mid-conversation: "Focus on China from now on"
- Pinned entities always included in BFS context
- Visual indicator in graph: pinned nodes glow cyan

#### Source Attribution (Roadmap)
- Every LLM answer cites the specific graph edges that informed it
- Click a citation → graph highlights those edges

#### Analyst Personas (Roadmap)
```
PERSONA: "You are a hawkish US defense analyst..."
PERSONA: "You are a UN conflict mediator..."  
PERSONA: "You are a Chinese foreign policy strategist..."
→ Same graph, radically different interpretations
```

---

## 9. Frontend — Command Interface

### 9.1 Design Language
**"Classified Cartography"** — dark intelligence terminal aesthetic.

```
Fonts:   Syne Mono (headers) · DM Mono (data) · Barlow Condensed (body)
Colors:  #050810 background · #00d4ff cyan · #ffb347 amber · #00ff99 green
         #ff3355 red · #c084fc purple · #ff6b35 orange
```

### 9.2 Layout Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          HUD (52px)                              │
│  ◈GOIES  [STATUS]  [NODES] [EDGES]  [RISK BADGE]  [GRAPH|GEO]  │
│           [MODEL ▾]  [JSON] [CSV]  [CLEAR]                      │
├─────┬───────────────────────────────────────────────────┬───────┤
│     │                                                   │       │
│  I  │          Full-screen vis.js graph                │  A    │
│  N  │          or Leaflet world map                    │  N    │
│  G  │                                                   │  A    │
│  E  │          dot-grid background                     │  L    │
│  S  │          scanline overlay                        │  Y    │
│  T  │          vignette                                │  S    │
│     │                                                   │  T    │
├─────┴───────────────────────────────────────────────────┴───────┤
│                   Node Inspector (slides up)                     │
└─────────────────────────────────────────────────────────────────┘
[⚡INGEST]        [◈SIMULATE]  [⚠FORECAST]              [💬ANALYST]
```

### 9.3 Planned UI Features

#### Timeline Slider (Roadmap)
```
HUD bottom strip:
◀ [────●─────────────────] ▶  |  2026-03-01 → 2026-03-09
```
- Scrub between graph snapshots
- Auto-play animation mode
- Per-snapshot entity count badge

#### Report Generator Panel (Roadmap)
```
[📄 GENERATE BRIEF]
  → Select entities to include
  → Select date range
  → Choose format: PDF / DOCX / Markdown
  → LLM drafts executive summary + key findings
  → Download
```

#### Alert Watch Panel (Roadmap)
```
[🔔 WATCH LIST]
  + Taiwan (trigger: tension > 60)
  + Iran (trigger: new hostile edge)
  + North Korea (trigger: any new connection)
```

#### ⌘K Command Palette — Expanded (Roadmap)
```
⌘K opens: search nodes + run commands
  > simulate: US lifts Iran sanctions
  > forecast
  > path: Russia → Germany
  > watch: Taiwan
  > export: json
  > clear
  > snapshot
```

---

## 10. Data Persistence & Storage

### 10.1 Current (File-Based)

```
goies_graph.json          # NetworkX node-link JSON
goies_snapshots/          # Timestamped graph versions (roadmap)
  2026-03-01T10:00:00.json
  2026-03-05T14:30:00.json
  2026-03-09T11:00:00.json
```

### 10.2 Graph JSON Schema

```json
{
  "directed": true,
  "multigraph": false,
  "graph": {},
  "nodes": [
    {
      "id": "Russia",
      "group": "country",
      "confidence": 0.94,
      "attributes": {"capital": "Moscow"},
      "ingested_at": "2026-03-09T11:00:00Z"
    }
  ],
  "links": [
    {
      "source": "Russia",
      "target": "Ukraine",
      "label": "invades",
      "confidence": 0.97,
      "ingested_at": "2026-03-09T11:00:00Z"
    }
  ]
}
```

### 10.3 Export Formats

| Format | Endpoint | Use Case |
|--------|----------|----------|
| JSON | `GET /api/export/json` | Full graph backup, re-import |
| CSV | `GET /api/export/csv` | Edge list for Excel/Pandas |
| GraphML | `GET /api/export/graphml` | Gephi, Cytoscape, yEd |
| PDF Brief | `GET /api/export/brief` | (Roadmap) Executive intelligence report |

### 10.4 Future: SQLite Backend (Roadmap)

```sql
-- entities table
CREATE TABLE entities (
    id TEXT PRIMARY KEY,
    group_type TEXT,
    confidence REAL,
    attributes JSON,
    ingested_at TIMESTAMP,
    source_count INTEGER DEFAULT 1,
    tension_score REAL DEFAULT 0
);

-- relationships table  
CREATE TABLE relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_entity TEXT REFERENCES entities(id),
    to_entity TEXT REFERENCES entities(id),
    label TEXT,
    confidence REAL,
    ingested_at TIMESTAMP,
    source TEXT
);

-- snapshots table
CREATE TABLE snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP,
    graph_json TEXT,
    note TEXT
);
```

---

## 11. Multi-Model & LLM Layer

### 11.1 Current
- Single Ollama endpoint at `localhost:11434`
- Model selectable via HUD dropdown
- Supports any model installed via `ollama pull`

### 11.2 Model Routing (Roadmap)

```python
# config.py
MODEL_ROUTES = {
    "extraction": "llama3.2",          # fast, structured JSON
    "simulation": "llama3.2:70b",      # deeper reasoning
    "forecasting": "llama3.2:70b",     # complex geopolitical analysis
    "chat": "llama3.2",               # conversational
    "report": "gemma2:27b",           # long-form writing
}
```

### 11.3 Extraction Prompt Engineering

```
System: You are a geopolitical entity extractor. Output ONLY valid JSON.

Schema:
{
  "entities": [{"id": string, "group": enum, "confidence": float, "attributes": object}],
  "relationships": [{"from": string, "to": string, "label": string, "confidence": float}]
}

Rules:
- confidence: 0.0–1.0 (drop <0.5)
- group: country | person | organization | technology | event | treaty | resource
- id: use canonical names (prefer "US" over "The United States of America")
- attributes: max 3 key-value pairs, factual only

TEXT: {text}
```

### 11.4 Prompt Versioning (Roadmap)

```
prompts/
  extraction_v1.txt    # original
  extraction_v2.txt    # with attribute extraction
  extraction_v3.txt    # current production
  simulation_v1.txt
  forecast_v1.txt
```

- A/B test prompts against extraction quality metrics
- Roll back to prior prompt version if quality degrades

---

## 12. Security & Access Control

### 12.1 Current
- No authentication (single-user local app)
- All data stays local
- No telemetry, no external API calls except Ollama

### 12.2 Multi-User Mode (Roadmap)

```python
# auth.py
class User(BaseModel):
    id: str
    name: str
    role: Literal["admin", "analyst", "viewer"]
    graph_access: List[str]   # which named graphs they can access

# Named graphs
GET  /api/graphs              # list all graphs
POST /api/graphs/{name}       # create graph
GET  /api/graphs/{name}/nodes # query specific graph
```

### 12.3 API Key Authentication (Roadmap)

```python
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if request.url.path.startswith("/api"):
        key = request.headers.get("X-API-Key")
        if not verify_key(key):
            return Response(status_code=401)
    return await call_next(request)
```

### 12.4 Data Classification (Roadmap)

```python
# Node-level classification
{
  "id": "Operation Northstar",
  "group": "event",
  "classification": "SENSITIVE",   # PUBLIC | SENSITIVE | RESTRICTED
  "visible_to": ["admin", "analyst"]
}
```

---

## 13. Observability & Monitoring

### 13.1 Logging

```python
# Production logging setup
import structlog

log = structlog.get_logger()

# Extraction events
log.info("extraction.complete", 
         entities=12, relationships=8, 
         model="llama3.2", duration_ms=4200)

# Simulation events
log.info("simulation.complete",
         scenario="US lifts Iran sanctions",
         risk_score=38, risk_label="MEDIUM",
         duration_ms=8100)
```

### 13.2 Health Endpoint

```json
GET /api/health
{
  "status": "healthy",
  "ollama": {"online": true, "models": ["llama3.2", "gemma2"]},
  "graph": {"nodes": 142, "edges": 287, "components": 3},
  "uptime_seconds": 3600,
  "version": "3.0.0"
}
```

### 13.3 Metrics (Roadmap)

```
/api/metrics  →  Prometheus-compatible

goies_extractions_total
goies_extraction_duration_ms
goies_simulations_total
goies_graph_nodes
goies_graph_edges
goies_ollama_latency_ms
```

---

## 14. Testing Strategy

### 14.1 Unit Tests

```
tests/
  test_extractor.py     # entity extraction accuracy
  test_geo.py           # tension score calculations
  test_simulator.py     # simulation doesn't mutate live graph
  test_forecaster.py    # structural signal detection
  test_utils.py         # chunking, fuzzy resolution, analytics
```

#### Key Test Cases

```python
# test_simulator.py
def test_simulation_does_not_mutate_live_graph():
    """CRITICAL: simulation must never modify the live graph."""
    graph = build_test_graph()
    original_edges = set(graph.edges())
    run_simulation("US lifts Iran sanctions", graph)
    assert set(graph.edges()) == original_edges

# test_geo.py
def test_hostile_edges_increase_tension():
    graph = nx.DiGraph()
    graph.add_node("Russia", group="country")
    graph.add_node("Ukraine", group="country")
    graph.add_edge("Russia", "Ukraine", label="invades")
    tensions = calculate_country_tensions(graph)
    assert tensions["Russia"] > tensions["Ukraine"]   # aggressor > target

# test_utils.py
def test_fuzzy_resolution_deduplicates():
    graph = nx.DiGraph()
    graph.add_node("United States", group="country")
    assert resolve_node_name(graph, "US") == "United States"
    assert resolve_node_name(graph, "USA") == "United States"
```

### 14.2 Integration Tests

```python
# tests/test_api.py
async def test_extract_endpoint():
    response = await client.post("/api/extract", json={
        "text": "Russia invaded Ukraine in February 2022.",
        "model": "llama3.2"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["entities"] > 0
    assert data["relations"] > 0
    assert "Russia" in [n["id"] for n in data["vis"]["nodes"]]
```

### 14.3 Prompt Quality Testing

```python
# tests/test_prompts.py
GOLD_STANDARD = [
    {
        "text": "The US imposed sanctions on Russia following the Ukraine invasion.",
        "expected_entities": ["US", "Russia", "Ukraine"],
        "expected_relationships": [("US", "Russia", "sanctions")]
    },
]

def test_extraction_accuracy():
    for case in GOLD_STANDARD:
        result = extract_intelligence(case["text"])
        entity_ids = {e.extraction_text for e in result}
        assert case["expected_entities"][0] in entity_ids  # at minimum
```

---

## 15. Deployment

### 15.1 Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start Ollama
ollama pull llama3.2
ollama run llama3.2   # keep running in separate terminal

# Start GOIES
uvicorn server:app --reload --port 8000 --host 0.0.0.0

# Open
open http://localhost:8000
```

### 15.2 GitHub Pages (Static Mode)

```bash
# Just replace index.html in the repo root
cp build/index.html index.html
git add index.html
git commit -m "deploy: update GOIES SPA"
git push origin main

# Ollama must be running locally with CORS enabled
OLLAMA_ORIGINS="https://your-username.github.io" ollama serve
```

### 15.3 Docker (Roadmap)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  goies:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
  
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

volumes:
  ollama_data:
```

### 15.4 requirements.txt

```
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
networkx>=3.2
pyvis>=0.3.2
requests>=2.31.0
python-multipart>=0.0.9   # for file upload endpoints
httpx>=0.27.0             # for URL ingestion (roadmap)
feedparser>=6.0.11        # for RSS ingestion (roadmap)
pypdf2>=3.0.1             # for PDF ingestion (roadmap)
python-docx>=1.1.0        # for DOCX ingestion (roadmap)
reportlab>=4.1.0          # for PDF report generation (roadmap)
```

---

## 16. Feature Backlog

### P0 — Critical / Next Sprint

| Feature | Module | Effort | Impact |
|---------|--------|--------|--------|
| Streaming extraction (SSE) | extractor.py | M | High — UX dramatically better |
| URL scraping ingestion | ingestor.py | M | High — removes copy-paste friction |
| Graph snapshot versioning | utils.py + server.py | M | High — enables timeline view |

### P1 — High Value

| Feature | Module | Effort | Impact |
|---------|--------|--------|--------|
| Path finder UI | graph_algo.py | M | High — core analyst workflow |
| Timeline slider | frontend | L | High — temporal intelligence |
| Entity merge UI | frontend | M | Medium — graph hygiene |
| PDF/DOCX upload | ingestor.py | S | High — removes copy-paste friction |
| Conflict detection | utils.py | S | Medium — data quality signal |

### P2 — Valuable Enhancements

| Feature | Module | Effort | Impact |
|---------|--------|--------|--------|
| Report generator (PDF brief) | reporter.py | L | High — analyst deliverable |
| Tension alert watch list | geo.py + frontend | M | Medium — proactive monitoring |
| Simulation history + compare | simulator.py | M | Medium — audit trail |
| Analyst personas | server.py | S | Medium — analytical depth |
| Source attribution in chat | server.py | M | Medium — trust/transparency |
| Choropleth tension map | frontend | M | Medium — better geo UX |
| RSS auto-poll ingestion | ingestor.py | M | Medium — always-on monitoring |

### P3 — Future Research

| Feature | Description |
|---------|-------------|
| Multi-user graphs | Named graphs with role-based access |
| Monte Carlo simulation | 10× runs with risk distribution |
| Historical calibration | Brier score tracking for forecasts |
| Twitter/Nitter ingestion | Social media OSINT layer |
| Graph federation | Share/merge graphs between analysts |
| Fine-tuned extraction model | Domain-specific llama fine-tune on geopolitical corpora |

---

## 17. File Structure

```
GOIES/
├── index.html              # Standalone GitHub Pages SPA (all-in-one)
│
├── server.py               # FastAPI backend (217 lines)
│   ├── GET  /api/health
│   ├── GET  /api/models
│   ├── GET  /api/graph
│   ├── POST /api/extract
│   ├── POST /api/query
│   ├── POST /api/simulate
│   ├── POST /api/forecast
│   ├── GET  /api/geo
│   ├── DELETE /api/graph
│   └── GET  /api/export/{fmt}
│
├── extractor.py            # Ollama extraction engine (175 lines)
│   ├── extract_intelligence(text, model)
│   ├── check_ollama_health()
│   └── list_available_models()
│
├── utils.py                # Shared utilities (172 lines)
│   ├── chunk_text()
│   ├── resolve_node_name()   # fuzzy deduplication
│   ├── save_graph() / load_graph()
│   ├── get_graph_analytics()
│   ├── get_ego_subgraph()
│   ├── retrieve_graph_context()
│   ├── export_json() / export_csv() / export_graphml()
│
├── geo.py                  # Geo-positional tension engine (210 lines)
│   ├── COUNTRY_COORDS        # 80+ countries
│   ├── calculate_country_tensions()
│   ├── normalise_tensions()
│   └── get_geo_data()
│
├── simulator.py            # Policy simulation (274 lines)
│   ├── run_simulation(scenario, graph, model)
│   ├── _parse_scenario()     # LLM pass 1
│   ├── _apply_changes()      # graph clone mutation
│   └── _cascade_analysis()   # LLM pass 2
│
├── forecaster.py           # Crisis forecasting (281 lines)
│   ├── run_forecast(graph, model)
│   ├── _structural_signals() # hotspots, reciprocal, triangles
│   └── _hotspot_nodes()
│
├── static/
│   └── index.html          # Full SPA (served by FastAPI)
│
├── requirements.txt
├── .gitignore
└── PROJECT.md              # This file
```

---

## 18. API Reference

### POST /api/extract

```json
Request:
{
  "text": "Russia imposed sanctions on Ukraine...",
  "model": "llama3.2"
}

Response:
{
  "extractions": 18,
  "entities": 10,
  "relations": 8,
  "nodes_added": 4,
  "edges_added": 6,
  "new_node_ids": ["Russia", "Ukraine", "EU", "Zelensky"],
  "vis": { "nodes": [...], "edges": [...] },
  "analytics": {
    "nodes": 42, "edges": 87, "density": 0.0502,
    "weakly_connected_components": 3,
    "group_counts": {"country": 12, "person": 8, ...},
    "top_degree": [["Russia", 0.82], ...],
    "top_betweenness": [["US", 0.74], ...]
  }
}
```

### POST /api/simulate

```json
Request:
{ "scenario": "US lifts all sanctions on Iran", "model": "llama3.2" }

Response:
{
  "scenario": "US lifts all sanctions on Iran",
  "risk_score": 38.5,
  "risk_label": "MEDIUM",
  "cascade_narrative": "The removal of US sanctions on Iran would fundamentally alter...",
  "second_order": [
    "Israel likely to escalate military exercises near Iranian border",
    "Gulf states reassess US security guarantees",
    "Iranian oil exports resume, depressing global crude prices by 8–12%"
  ],
  "added_edges": [{"from": "US", "to": "Iran", "label": "diplomatic talks"}],
  "removed_edges": [{"from": "US", "to": "Iran", "label": "sanctions"}],
  "affected_nodes": ["US", "Iran", "Israel", "Saudi Arabia", "IAEA"],
  "model_used": "llama3.2"
}
```

### POST /api/forecast

```json
Request:
{ "model": "llama3.2", "focus": "Middle East" }

Response:
{
  "global_risk": 67.0,
  "global_label": "HIGH",
  "structural_summary": "The intelligence graph shows 3 mutual hostility pairs...",
  "hotspot_nodes": ["Russia", "Iran", "North Korea", "China", "Israel"],
  "forecasts": [
    {
      "rank": 1,
      "title": "Taiwan Strait Military Crisis",
      "actors": ["China", "Taiwan", "US"],
      "probability": 0.73,
      "severity": "CRITICAL",
      "timeframe": "near-term (0-3 months)",
      "structural_signal": "Reciprocal hostility between China and Taiwan with US as high-betweenness broker",
      "narrative": "Structural analysis reveals China and Taiwan in mutual hostile relationship...",
      "mitigation": "Activate US-China hotline; ASEAN mediation track recommended."
    }
  ],
  "model_used": "llama3.2"
}
```

### GET /api/geo

```json
Response:
{
  "markers": [
    {
      "id": "Russia",
      "lat": 61.52, "lon": 105.31,
      "group": "country",
      "tension": 84.3,
      "raw_score": 142.6,
      "color": "#ff2244",
      "connections": ["invades Ukraine", "threatens NATO", "sanctions EU"],
      "degree": 18,
      "attributes": "{\"capital\": \"Moscow\"}"
    }
  ],
  "total": 14
}
```

---

*Last updated: March 9, 2026 · GOIES v3*  
*Maintained by: tanu-1403*