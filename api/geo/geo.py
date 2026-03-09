"""
geo.py — GOIES Geo-Positional Intelligence Layer
Computes per-country tension metrics from the knowledge graph and maps
entities to real-world coordinates for the world-map view.

Tension score (0–100) is derived from:
  - Hostile edge labels (sanctions, attacks, threatens, invades…)  → +high
  - Neutral/diplomatic edges                                        → +low
  - Incoming hostile pressure                                       → +medium
  - Degree centrality in conflict clusters                          → multiplicative
  - Connected 'Event' nodes of type military/conflict               → +medium
Score is normalised globally so the most tense country = 100.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

# ── Country centroids (lat, lon) — ~80 countries ─────────────────────────────
COUNTRY_COORDS: Dict[str, Tuple[float, float]] = {
    # Major powers
    "United States": (37.09, -95.71),
    "US": (37.09, -95.71),
    "USA": (37.09, -95.71),
    "China": (35.86, 104.19),
    "Russia": (61.52, 105.31),
    "Germany": (51.16, 10.45),
    "France": (46.23, 2.21),
    "United Kingdom": (55.37, -3.43),
    "UK": (55.37, -3.43),
    "Japan": (36.20, 138.25),
    "India": (20.59, 78.96),
    "Brazil": (-14.24, -51.93),
    "Canada": (56.13, -106.35),
    "Australia": (-25.27, 133.77),
    "South Korea": (35.90, 127.77),
    "North Korea": (40.33, 127.51),
    "Iran": (32.42, 53.68),
    "Israel": (31.04, 34.85),
    "Saudi Arabia": (23.88, 45.08),
    "Turkey": (38.96, 35.24),
    "Pakistan": (30.37, 69.34),
    "Afghanistan": (33.93, 67.71),
    "Ukraine": (48.37, 31.16),
    "Poland": (51.91, 19.14),
    "Taiwan": (23.69, 120.96),
    # Middle East & Africa
    "Iraq": (33.22, 43.67),
    "Syria": (34.80, 38.99),
    "Lebanon": (33.85, 35.86),
    "Jordan": (30.58, 36.24),
    "Egypt": (26.82, 30.80),
    "Libya": (26.33, 17.22),
    "Yemen": (15.55, 48.52),
    "UAE": (23.42, 53.85),
    "Qatar": (25.35, 51.18),
    "Kuwait": (29.31, 47.48),
    "Bahrain": (25.93, 50.63),
    "Oman": (21.51, 55.92),
    "Nigeria": (9.08, 8.67),
    "South Africa": (-30.56, 22.94),
    "Ethiopia": (9.14, 40.49),
    "Kenya": (-0.02, 37.90),
    "Somalia": (5.15, 46.20),
    "Sudan": (12.86, 30.22),
    "Algeria": (28.03, 1.66),
    "Morocco": (31.79, -7.09),
    "Tunisia": (33.89, 9.54),
    # Europe
    "Italy": (41.87, 12.56),
    "Spain": (40.46, -3.74),
    "Netherlands": (52.13, 5.29),
    "Belgium": (50.50, 4.47),
    "Sweden": (60.13, 18.64),
    "Norway": (60.47, 8.47),
    "Finland": (61.92, 25.75),
    "Denmark": (56.26, 9.50),
    "Switzerland": (46.82, 8.23),
    "Austria": (47.52, 14.55),
    "Czech Republic": (49.82, 15.47),
    "Romania": (45.94, 24.96),
    "Hungary": (47.16, 19.50),
    "Greece": (39.07, 21.82),
    "Serbia": (44.01, 21.00),
    "Belarus": (53.71, 27.95),
    # Asia
    "Indonesia": (-0.79, 113.92),
    "Vietnam": (14.06, 108.28),
    "Philippines": (12.88, 121.77),
    "Thailand": (15.87, 100.99),
    "Malaysia": (4.21, 101.97),
    "Singapore": (1.35, 103.82),
    "Bangladesh": (23.68, 90.35),
    "Myanmar": (17.10, 96.00),
    "Sri Lanka": (7.87, 80.77),
    "Nepal": (28.39, 84.12),
    "Kazakhstan": (48.02, 66.92),
    "Uzbekistan": (41.38, 64.59),
    # Americas
    "Mexico": (23.63, -102.55),
    "Argentina": (-38.41, -63.62),
    "Colombia": (4.57, -74.30),
    "Venezuela": (6.42, -66.59),
    "Cuba": (21.52, -79.97),
    "Chile": (-35.68, -71.54),
    "Peru": (-9.19, -75.02),
    # Pacific
    "New Zealand": (-40.90, 174.89),
    # Blocs / Regions (approximate centres)
    "EU": (50.00, 10.00),
    "NATO": (51.50, 10.00),
    "ASEAN": (5.00, 110.00),
    "West Asia": (27.00, 43.00),
    "Gulf": (25.00, 51.00),
    "Africa": (8.78, 34.51),
    "Europe": (54.53, 15.25),
    "South Asia": (20.00, 77.00),
    "Southeast Asia": (10.00, 106.00),
    "Latin America": (-15.00, -60.00),
    "Central Asia": (45.00, 68.00),
}

# ── Hostility vocabulary ──────────────────────────────────────────────────────
HIGH_HOSTILE = re.compile(
    r"sanction|attack|invad|bomb|missile|strike|kill|assassin|"
    r"threaten|wage war|blockade|seize|destabil|terroris|deploy troops",
    re.I,
)
MED_HOSTILE = re.compile(
    r"restrict|ban|expel|arrest|accuse|confront|dispute|protest|"
    r"tension|pressure|isolat|impound|cyber",
    re.I,
)
COOPERATIVE = re.compile(
    r"cooperat|partner|ally|invest|aid|support|trade|agree|pact|"
    r"treaty|collaborat|assist|negotiate",
    re.I,
)


def _edge_score(label: str) -> float:
    """Returns a signed tension contribution for a single edge label."""
    if HIGH_HOSTILE.search(label):
        return 18.0
    if MED_HOSTILE.search(label):
        return 9.0
    if COOPERATIVE.search(label):
        return -3.0  # cooperation slightly reduces tension
    return 2.0  # neutral/unknown edges add a small baseline


def calculate_country_tensions(graph: nx.DiGraph) -> Dict[str, float]:
    """
    Compute raw tension scores for every Country/Organization node.
    Returns a dict {node_name: raw_score}.
    """
    scores: Dict[str, float] = {}

    for node, data in graph.nodes(data=True):
        group = data.get("group", "unknown").lower()
        if group not in ("country", "organization", "unknown"):
            continue

        score = 0.0

        # Outgoing hostile edges
        for _, _, edata in graph.out_edges(node, data=True):
            label = edata.get("label", "")
            s = _edge_score(label)
            score += s * (1.2 if s > 0 else 0.8)  # aggressors score higher

        # Incoming hostile pressure
        for _, _, edata in graph.in_edges(node, data=True):
            label = edata.get("label", "")
            score += _edge_score(label) * 0.9

        # Connected Event nodes of type military/conflict
        for neighbour in list(graph.predecessors(node)) + list(graph.successors(node)):
            ndata = graph.nodes.get(neighbour, {})
            if ndata.get("group") == "event":
                attrs_str = ndata.get("title", "").lower()
                if any(
                    w in attrs_str
                    for w in ("military", "conflict", "war", "attack", "crisis")
                ):
                    score += 7.0

        # Boost by degree centrality (well-connected nodes feel more tension)
        degree = graph.degree(node)
        score *= 1.0 + 0.04 * min(degree, 20)

        scores[node] = max(0.0, score)

    return scores


def normalise_tensions(raw: Dict[str, float]) -> Dict[str, float]:
    """Normalise raw scores to 0–100. Returns empty dict if no data."""
    if not raw:
        return {}
    max_val = max(raw.values()) or 1.0
    return {k: round(min(100.0, v / max_val * 100), 1) for k, v in raw.items()}


def tension_color(score: float) -> str:
    """Returns a hex color for a 0–100 tension score."""
    if score >= 75:
        return "#ff2244"  # critical red
    if score >= 50:
        return "#ff6b35"  # high orange
    if score >= 25:
        return "#ffaa40"  # medium amber
    if score >= 10:
        return "#ffe066"  # low yellow
    return "#00ff88"  # peaceful green


def get_geo_data(graph: nx.DiGraph) -> List[Dict[str, Any]]:
    """
    Builds the full geo-layer payload for the frontend.
    Returns a list of country markers with coordinates, tension, and connections.
    """
    raw = calculate_country_tensions(graph)
    normed = normalise_tensions(raw)

    result: List[Dict[str, Any]] = []

    for node, data in graph.nodes(data=True):
        group = data.get("group", "unknown").lower()

        # Only include nodes that have coordinates
        coords = COUNTRY_COORDS.get(node)
        if not coords:
            # Try partial match (e.g. "United States" matches "US")
            for alias, c in COUNTRY_COORDS.items():
                if alias.lower() in node.lower() or node.lower() in alias.lower():
                    coords = c
                    break

        if not coords:
            continue

        tension = normed.get(node, 0.0)
        raw_score = raw.get(node, 0.0)

        # Build connection list for tooltip
        connections: List[str] = []
        for u, v, edata in graph.edges(data=True):
            if u == node or v == node:
                other = v if u == node else u
                connections.append(f"{edata.get('label','→')} {other}")

        result.append(
            {
                "id": node,
                "lat": coords[0],
                "lon": coords[1],
                "group": group,
                "tension": tension,
                "raw_score": round(raw_score, 1),
                "color": tension_color(tension),
                "connections": connections[:8],
                "degree": graph.degree(node),
                "attributes": data.get("title", "{}"),
            }
        )

    return result
