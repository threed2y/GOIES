# GOIES: Geopolitical Open Intelligence & Extraction System

GOIES is a comprehensive, AI-powered geopolitical intelligence platform. It ingests raw intelligence feeds—ranging from news articles and PDF reports to word documents—extracts structured entities and relationships using a Large Language Model (LLM), and maps them into an interactive, mathematical knowledge graph.

## Architecture

GOIES is built as a Single Page Application (SPA) powered by a lightweight Python backend.
- **Frontend**: Vanilla HTML/JS/CSS utilizing `vis.js` for graph rendering and `Chart.js` for analytics. No heavy Node or Webpack builds required.
- **Backend / API**: Python 3 based API powered by `FastAPI` and `uvicorn`.
- **Logic & Intelligence**: Powered by `NetworkX` for graph topology calculations and `Ollama` (running `llama3.2`) for running local generative AI extractions and analysis.

## Core Features

1. **LLM Intelligence Extraction (`extractor.py`)**  
   GOIES consumes unstructured text, securely passing it to a local LLM to extract Actors (countries, people, organizations) and Relationships (hostile, cooperative, neutral). The extraction output organically streams directly onto the graph canvas.

2. **Graph Topology Engine (`utils.py`, `graph_algo.py`)**  
   The platform measures the "influence" and "vulnerability" of nodes automatically:
   - Calculate Betweenness Centrality, Degree Centrality, and Graph Density.
   - Run Pathfinding analysis to discover the shortest route of influence between two geopolitical actors.

3. **Geopolitical Risk & Tension (`geo.py`)**  
   Dynamically assess global flashpoints. The engine reads relationships (e.g., "sanctions," "attacks") and computes 0-100 tension risk scores per country. Users can define custom alert thresholds.

4. **Watch List & Real-Time Alerts**  
   Analysts can add nodes to a dedicated Watch List with user-defined tripwires. If a country's risk score spikes above the threshold due to new ingestion, the dashboard fires an alert.

5. **Strategic Event Simulation (`simulator.py`)**  
   Analysts can simulate hypothetical events (e.g., "Coup in Country X") and watch the LLM recalculate relationship weights, infer cascading disruptions, and update the graph in real-time. The **Compare Mode** lets you contrast the structural impacts of two different simulation runs side by side.

6. **Graph Health & Automatic Summaries**  
   - **Health Score**: GOIES tracks the statistical richness (node diversity, relationship density) of your graph—ensuring you don't form biases over sparse extractions.
   - **Strategic Summaries**: One click queries the LLM to write a live, 3-paragraph executive briefing of the entire state of the geopolitical map.

7. **Report Generator (`reporter.py`)**  
   Need to leave the terminal? Generate dynamic intelligence briefings focusing on specific entities. Complete with LLM-curated dossiers and auto-rendered PDF or Markdown execution formats.

## Quick Start
1. **Ensure Ollama is running (`llama3.2`)**  
   ```bash
   ollama run llama3.2
   ```
2. **Setup Python environment and dependencies**  
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Start the FastAPI Server**  
   ```bash
   uvicorn server:app --reload --host 0.0.0.0 --port 8000
   ```
4. **Access Dashboard**  
   Open `http://localhost:8000/index.html` in your browser.

## Docker Deployment (Recommended)
You can deploy GOIES and the Ollama LLM service natively via Docker Compose:

1. **Start the containers**
   ```bash
   docker-compose up -d --build
   ```
2. **Initialize the LLM inside the Ollama container (One-time only)**
   ```bash
   docker exec -it goies-ollama ollama run llama3.2
   ```
   *(Wait for the model to finish pulling before interacting with the extraction features)*

3. **Access Dashboard**
   Navigate to `http://localhost:8000/index.html`

## Repository Layout
- `server.py`: The FastAPI server. Handles `/api/*` endpoints and serves the static frontend.
- `index.html`: The all-in-one frontend dashboard.
- `extractor.py`: The LLM interface feeding intelligence into GOIES.
- `geo.py`, `utils.py`, `graph_algo.py`: The network analytics & geopolitical scoring engines.
- `reporter.py`: Handles exporting PDF and MD profiles.
- `simulator.py`, `forecaster.py`: The strategic analysis and predictive features.
- `Dockerfile` & `docker-compose.yml`: Containerization and service orchestration.
