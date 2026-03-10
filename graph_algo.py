"""
graph_algo.py — Algorithms for GOIES
Provides path finding and advanced graph analytics.
"""

import networkx as nx
from typing import Dict, Any, List

def find_shortest_path(graph: nx.DiGraph, from_node: str, to_node: str, max_depth: int = 4) -> Dict[str, Any]:
    """
    Finds the shortest directed path between two nodes.
    Returns the ordered list of nodes and the edges connecting them.
    Currently uses simple shortest path. Ignores max_depth if > shortest.
    Returns empty structures if no path exists.
    """
    if from_node not in graph or to_node not in graph:
        return {"nodes": [], "edges": []}
        
    try:
        # Using shortest path. In a directed graph, this respects edge directions.
        path_nodes = nx.shortest_path(graph, source=from_node, target=to_node)
        
        # Enforce max depth constraint
        if len(path_nodes) - 1 > max_depth:
            return {"nodes": [], "edges": []}
            
        # Collect edge data for the path
        path_edges = []
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i+1]
            edge_data = graph.get_edge_data(u, v)
            if edge_data:
                # networkx returns a dict of attributes for the edge
                # If there are multiple edges (MultiDiGraph), it's nested, but our schema is DiGraph
                path_edges.append({
                    "from": u,
                    "to": v,
                    "label": edge_data.get("label", ""),
                    "confidence": edge_data.get("confidence", 1.0)
                })
                
        return {
            "nodes": path_nodes,
            "edges": path_edges
        }
    except nx.NetworkXNoPath:
        return {"nodes": [], "edges": []}
