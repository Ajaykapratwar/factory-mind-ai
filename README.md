# FactoryMind Pharma

FactoryMind Pharma is an end-to-end data pipeline and dashboard infrastructure project for forecasting pharmaceutical demand.

## Project Structure

- `factorymind_pharma/` - Main application code
  - `run_pipeline.py` - End-to-end pipeline runner
  - `dashboard/` - Dash application for Executive and Operational dashboards
  - `services/` - Core logic for data sensing and forecasting
  - `evaluation/` - Metrics and evaluation suite
  - `configs/` - Project configuration files
- `data/` - Raw, processed, and output data directories
- `venv/` - Virtual environment (git-ignored)

## Setup Instructions

1. **Activate your virtual environment**
   - **PowerShell:** `.\venv\Scripts\Activate.ps1`
   - **Command Prompt:** `venv\Scripts\activate.bat`

2. **Install requirements**
   ```bash
   pip install -r factorymind_pharma/requirements.txt
   ```

## Usage

### 1. Run the Data Pipeline
Processes the raw data, engineers features, trains the forecasting models, and generates evaluation metrics.
```bash
python factorymind_pharma/run_pipeline.py
```

### 2. Run the Dashboard
Starts the multi-page Dash web application.
```bash
python factorymind_pharma/dashboard/app.py
```
By default, the dashboard is served at `http://localhost:8050`.
- **Executive Summary:** `http://localhost:8050/`
- **Operational Planning:** `http://localhost:8050/operational`
