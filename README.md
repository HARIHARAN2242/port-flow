================================================================================
⚓ PORTFLOW — SMART MARITIME & PORT OPERATIONS
================================================================================

PortFlow is an intelligent maritime logistics and port operations dashboard designed 
to bridge historical port-dwell benchmark data with practical, real-time terminal 
planning and decision support. 


--------------------------------------------------------------------------------
🏛️ SYSTEM ARCHITECTURE
--------------------------------------------------------------------------------

PortFlow follows a modular, multi-tier architecture separating the user interface, 
decision optimization logic, historical benchmark data processing, and local AI integration:

[ Streamlit Web Interface ] 
       │
       ├─► [ Benchmark Data Module ] ─────────► Historical Port-Dwell Benchmark Data (Zenodo)
       │
       ├─► [ Operational Optimizer (optimizer.py) ] ──► Truck Planning & Yard Calculations
       │
       ├─► [ GIS Mapping Layer (Folium) ] ──────► Port Locations & Live AIS / ships.csv Feed
       │
       └─► [ Local AI Assistant ] ─────────────► Ollama API (DeepSeek-R1 / Local LLM)


Core Architectural Components:
1. Interactive Presentation Layer (portflow_app.py): Built with Streamlit, it provides 
   a clean 3-tab layout for operations management, conversational AI guidance, and benchmark evaluation.
2. Historical Benchmark Data Module: Integrates Zenodo-hosted public port-dwell data 
   covering major US port complexes to establish baseline dwell times and risk indices.
3. Operational Optimization Logic (optimizer.py): Computes container-to-truck allocation 
   plans and slot requirements for regional vessel operations.
4. GIS Visualization Layer (streamlit-folium): Renders interactive maps mapping key regional 
   port coordinates alongside real or illustrative vessel tracking coordinates.
5. Local AI Remedy Assistant: Integrates with local Ollama instances (defaulting to DeepSeek 
   models) to provide context-aware operational troubleshooting and remedy steps.


--------------------------------------------------------------------------------
✨ KEY FEATURES & FUNCTIONALITY
--------------------------------------------------------------------------------

* 📊 Operations & Benchmark Dashboard:
  - Select regional demo vessels (e.g., Chennai, Ennore, Kattupalli, and Tuticorin) to 
    evaluate container volumes, speeds, and weather impacts.
  - Instantly calculate planning truck counts and review structured truck appointment schemas.
  - Query historical benchmark port data to compare median dwell days and relative dwell-risk indices.

* 🗺️ Interactive Ship & Port Mapping:
  - Visualizes regional port infrastructure using precise geographic coordinates.
  - Automatically ingests local ships.csv position feeds if provided, falling back to 
    illustrative demo vessel markers when live data is absent.

* 🤖 AI Remedy Assistant:
  - A conversational sidekick powered by a local LLM via Ollama.
  - Grounded with strict system guardrails to provide safe, actionable, and structured 
    tactical responses without hallucinating live operational telemetry.

* 🧪 Data Analytics & Evaluation:
  - Inspect baseline historical metrics and record splits from Zenodo historical datasets.


--------------------------------------------------------------------------------
🚀 GETTING STARTED & INSTALLATION
--------------------------------------------------------------------------------

1. Prerequisites & Dependencies
Ensure you have Python installed, then install the required core packages:
   pip install streamlit pandas requests folium streamlit-folium

2. Configure Local AI (Optional)
To use the AI Remedy Assistant, ensure Ollama is installed and running locally with your desired model:
   - Default Endpoint: http://localhost:11434/api/chat
   - Default Model: deepseek-r1:1.5b
   *(You can customize these via OLLAMA_URL or OLLAMA_MODEL environment variables).*

3. Run the Application
Launch the Streamlit dashboard locally:
   streamlit run portflow_app.py


--------------------------------------------------------------------------------
📂 PROJECT FILE STRUCTURE
--------------------------------------------------------------------------------

* portflow_app.py   : Main Streamlit application entry point and UI layout.
* optimizer.py      : Yard and truck allocation calculation logic.
* ships.csv         : Optional local feed containing columns (vessel_name, latitude, 
                      longitude, current_port, speed_knots, destination) for real-time 
                      map plotting.


--------------------------------------------------------------------------------
⚠️ DISCLAIMER
--------------------------------------------------------------------------------
Public training benchmarks are derived from historical multi-port datasets (Zenodo) 
and serve as analytical benchmarks rather than verified real-time forecasting for 
localized terminal hubs. Regional demo vessels and calculations are structured for 
prototype planning and operational exploration.
