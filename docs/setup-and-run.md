# Setup And Run

## Prerequisites
- Python 3.12+
- PostgreSQL database reachable through connection URLs
- PowerShell or equivalent shell for local execution

## 1. Create The Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For test tooling:
```powershell
pip install -r requirements-dev.txt
```

## 2. Configure Environment Variables
Create `.env` from [.env.example](../.env.example).

Required:
- `DATABASE_URL_DIRECT`

Recommended:
- `DATABASE_URL_POOLED`

`pipeline/apply_sql.py` requires `DATABASE_URL_DIRECT`. The Streamlit app prefers `DATABASE_URL_POOLED` and falls back to `DATABASE_URL_DIRECT`.

## 3. Apply The Database Objects
```powershell
python pipeline\apply_sql.py
```

This applies:
- schema and base tables
- seed/reference data
- analytical views

## 4. Load Incident Data
```powershell
python pipeline\load_csv.py --path path\to\Incident_Management_CSV.csv
```

Supported flags:
```powershell
python pipeline\load_csv.py --path path\to\Incident_Management_CSV.csv --delimiter ';' --chunksize 5000
python pipeline\load_csv.py --path path\to\Incident_Management_CSV.csv --dry-run
python pipeline\load_csv.py --path path\to\Incident_Management_CSV.csv --truncate
```

Notes:
- input is expected to be semicolon-delimited by default
- timestamps are parsed day-first
- duplicate rows are skipped using computed `event_key` values
- `customer_satisfaction` is optional during ingestion

## 5. Launch The App
```powershell
streamlit run app.py
```

The app uses a shared global filter bar and six workstreams:
- `Command Center`
- `Flow`
- `Handoffs`
- `Quality`
- `Intake`
- `Enablement`

## 6. Run Verification
Syntax check:
```powershell
python -m py_compile app.py ui\theme.py ui\charts.py ui\primitives.py
```

Tests:
```powershell
python -m pytest -q
```

Optional integration tests:
- set `TEST_DATABASE_URL`
- rerun `pytest`

## Troubleshooting
- Missing DB config: verify `.env` or Streamlit secrets are set
- SQL verification failures: rerun `python pipeline\apply_sql.py` and inspect missing view or unmapped-event output
- Empty dashboard: confirm the loader inserted rows into `im.event_raw` and the SQL views are present
