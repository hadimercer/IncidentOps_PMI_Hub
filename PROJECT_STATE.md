# IncidentOps_PMI_Hub - Project State (Handoff)

## Current Stack
- Frontend/App: Streamlit (`app.py`)
- Language: Python
- Data/Charts: `pandas`, `plotly`
- DB client: `psycopg2` (Postgres)
- Config: `.env` via `python-dotenv` + `st.secrets`
- Database: Neon Postgres (prefers pooled URL, fallback direct)
- Caching: `st.cache_resource` (DB connection), `st.cache_data` (query results)

## Modules/Pages
- Executive Overview
- Process Explorer
- Bottlenecks
- Escalations & Handoffs
- Channel & Intake
- Knowledge & FCR
- Quality & CX
- Problem Candidates

## DB objects
Used views/tables in app queries:
- `im.v_case`
- `im.v_case_sla`
- `im.v_variant_summary`
- `im.v_transition_summary`
- `im.v_transition_by_variant`
- `im.v_dwell_by_event`
- `im.v_handoff_summary`
- `im.v_pingpong_cases`
- `im.v_worst_handoff_cases`
- `im.v_cx_summary`
- `im.v_cx_breakdown`
- `im.v_closure_compliance`
- `im.v_problem_candidates`
- `im.v_problem_candidate_cases`
- `im.v_problem_candidate_top_cases`
- `im.v_channel_summary`
- `im.v_channel_issue_summary`
- `im.v_fcr_cases`
- `im.v_fcr_summary`
- `im.v_kb_enablement_candidates`

## Deployment
- Platform: Streamlit Cloud
- App URL: `<STREAMLIT_CLOUD_URL_HERE>`
- Required secrets/env:
  - `DATABASE_URL_POOLED` (preferred)
  - `DATABASE_URL_DIRECT` (fallback)

## Known Issues
- If DB objects/views are missing or stale, page-level queries fail and show Streamlit errors.
- Neon cold starts can delay first successful query; connection retry/backoff is enabled.
- Legacy Neon URLs with `options=...` can trigger SNI/project mismatch; app now strips `options` from query params.

## Next 5 Tasks
1. Add a lightweight health check panel (DB reachable + key view row counts).
2. Add smoke tests for DB URL normalization and retry behavior.
3. Centralize SQL error handling to provide consistent user-facing error messages.
4. Add per-page load timing metrics to identify slow queries/views.
5. Document Streamlit Cloud deployment steps in `README.md` with secret setup screenshots/steps.

## Commands
```powershell
# setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# run locally
streamlit run app.py

# optional: syntax check
python -m py_compile app.py
```
