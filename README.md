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
