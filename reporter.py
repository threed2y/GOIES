"""
reporter.py — Intelligence Report Generator
Generates PDF briefs from the intelligence graph.
"""

import io
import networkx as nx
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

from utils import get_graph_analytics

def generate_report(graph: nx.DiGraph, focus_entities: list[str] = None, summary: str = "") -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = styles['Heading1']
    title_style.alignment = 1 # Center
    
    h2_style = styles['Heading2']
    body_style = styles['Normal']
    
    Story = []
    
    # Title
    Story.append(Paragraph("GOIES Intelligence Brief", title_style))
    Story.append(Spacer(1, 0.25 * inch))
    
    # Metadata
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    Story.append(Paragraph(f"<b>Generated:</b> {date_str}", body_style))
    if focus_entities:
        Story.append(Paragraph(f"<b>Focus Entities:</b> {', '.join(focus_entities)}", body_style))
    Story.append(Spacer(1, 0.2 * inch))

    # Executive Summary (LLM)
    if summary:
        Story.append(Paragraph("Executive Strategic Summary", h2_style))
        for para in summary.split('\n\n'):
            Story.append(Paragraph(para, body_style))
            Story.append(Spacer(1, 0.1 * inch))
        Story.append(Spacer(1, 0.2 * inch))
    
    # Analytics
    analytics = get_graph_analytics(graph)
    Story.append(Paragraph("Graph Analytics Overview", h2_style))
    Story.append(Paragraph(f"• Total Entities: {analytics['nodes']}", body_style))
    Story.append(Paragraph(f"• Total Relationships: {analytics['edges']}", body_style))
    Story.append(Paragraph(f"• Network Density: {analytics['density']}", body_style))
    Story.append(Spacer(1, 0.2 * inch))
    
    # Top Entities
    top_degree = analytics.get('top_degree', [])
    if top_degree:
        Story.append(Paragraph("Key Actors (By Connections)", h2_style))
        for n, score in top_degree:
            Story.append(Paragraph(f"• {n} ({score:.2f})", body_style))
        Story.append(Spacer(1, 0.2 * inch))
        
    # Conflicts
    conflicts = analytics.get('conflicts', [])
    if conflicts:
        Story.append(Paragraph("Detected Conflicts", h2_style))
        for c in conflicts[:10]: # Limit to top 10
            u, v = c['nodes']
            Story.append(Paragraph(f"• <b>{u}</b> and <b>{v}</b> have contradictory edges.", body_style))
        if len(conflicts) > 10:
            Story.append(Paragraph(f"...and {len(conflicts) - 10} more.", body_style))
        Story.append(Spacer(1, 0.2 * inch))
        
    # Bridge Nodes
    top_betweenness = analytics.get('top_betweenness', [])
    if top_betweenness:
        Story.append(Paragraph("Critical Bridge Nodes", h2_style))
        for n, score in top_betweenness:
            Story.append(Paragraph(f"• {n} ({score:.2f})", body_style))
            
    doc.build(Story)
    
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

def generate_markdown_report(graph: nx.DiGraph, focus_entities: list[str] = None, summary: str = "") -> str:
    lines = []
    lines.append("# GOIES Intelligence Brief\n")
    
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    lines.append(f"**Generated:** {date_str}")
    if focus_entities:
        lines.append(f"**Focus Entities:** {', '.join(focus_entities)}")
    lines.append("\n---\n")
    
    if summary:
        lines.append("## Executive Strategic Summary\n")
        lines.append(summary + "\n")
        lines.append("\n---\n")
        
    analytics = get_graph_analytics(graph)
    lines.append("## Graph Analytics Overview\n")
    lines.append(f"- Total Entities: {analytics['nodes']}")
    lines.append(f"- Total Relationships: {analytics['edges']}")
    lines.append(f"- Network Density: {analytics['density']:.4f}\n")
    
    top_degree = analytics.get('top_degree', [])
    if top_degree:
        lines.append("## Key Actors (By Connections)\n")
        for n, score in top_degree:
            lines.append(f"- **{n}** ({score:.2f})")
        lines.append("")
        
    conflicts = analytics.get('conflicts', [])
    if conflicts:
        lines.append("## Detected Conflicts\n")
        for c in conflicts[:10]:
            u, v = c['nodes']
            lines.append(f"- **{u}** and **{v}** have contradictory edges.")
        if len(conflicts) > 10:
            lines.append(f"- ...and {len(conflicts) - 10} more.\n")
        else:
            lines.append("")
            
    top_betweenness = analytics.get('top_betweenness', [])
    if top_betweenness:
        lines.append("## Critical Bridge Nodes\n")
        for n, score in top_betweenness:
            lines.append(f"- **{n}** ({score:.2f})\n")
            
    return "\n".join(lines)
