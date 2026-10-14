<<<<<<< HEAD
# PortFlow ML Upgrade

## What is included
- `ml_model.py`: supervised Random Forest regression for median port dwell time, with chronological holdout evaluation.
- `train_model.py`: trains and evaluates the model and saves `portflow_dwell_model.joblib`.
- `remedy_bot.py`: trained TF-IDF + Logistic Regression intent classifier and an operational remedy knowledge base.
- `app.py`: integrated Streamlit dashboard, remedy assistant, and model evaluation tab.
- `optimizer.py`: truck planning helper and cautious rerouting alert.

## Public dataset
Use the Zenodo dataset:
https://zenodo.org/records/21203605

Download `monthly_dwell_2015_2025.csv` and put it in the same folder as `app.py`. The file contains historical monthly vessel dwell statistics for Houston, LA/Long Beach, NY/NJ, Savannah and Seattle. It does **not** contain Tamil Nadu port records.

The code attempts to download the CSV automatically if it is missing, but manual download is recommended if your network blocks Zenodo.

Dataset columns used:
- `Port`
- `YearMonth`
- `UniqueVessels`
- `MedianDwellDays` (target)
- `TotalObservations`
- `AvgVesselLength`

## Install and run (Windows PowerShell)
```powershell
cd path\to\portflow_ml_upgrade
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

If PowerShell blocks activation, run:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train_model.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Groq API key (optional)
Do not hardcode API keys in `app.py`. Set an environment variable in PowerShell:
```powershell
$env:GROQ_API_KEY="your-new-key"
streamlit run app.py
```
Or configure Streamlit secrets in `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "your-new-key"
```
Never commit this file to a public repository. Since the previous uploaded app contained a hard-coded key, revoke/rotate that key before using the updated app.

## Model limitations (important)
- The Random Forest is trained on US port monthly records, not Tamil Nadu data.
- It predicts median dwell time in days, not vessel ETA delay.
- The displayed 0–100 dwell-risk index is a relative ranking based on each benchmark port's historical dwell distribution; it is not an official congestion percentage or berth occupancy.
- The existing Tamil Nadu demo vessel records remain demonstration inputs. The UI explicitly warns that US benchmark results are not validated Tamil Nadu forecasts.
- The remedy classifier is a real text ML pipeline, but the starter training examples are small and hand-labelled. Add many domain-specific examples and evaluate on unseen questions before claiming production-level accuracy.
- Do not use these predictions for live vessel routing or safety decisions without validation and operational oversight.

## To build a valid Tamil Nadu model
Collect time-stamped historical records for each target port, including observed dwell/turnaround time or actual arrival delay. Useful predictors include queue length, berth occupancy, number of vessels waiting, vessel type/size, cargo volume, weather observations, and timestamp. Train/test chronologically and report MAE/RMSE on a held-out period. Do not fabricate target labels or represent synthetic records as real.
=======
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
>>>>>>> 9f7907c45df0ebdf87a9fac90089305744023058
