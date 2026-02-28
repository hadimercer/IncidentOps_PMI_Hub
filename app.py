import os
import streamlit as st

st.set_page_config(page_title="IncidentOps PMI Hub", layout="wide")
st.title("IncidentOps PMI Hub")
st.caption("Process Mining Insights for Incident Management")

db_url = os.getenv("DATABASE_URL_POOLED") or os.getenv("DATABASE_URL_DIRECT")
if not db_url:
    st.warning("Database is not configured yet. Set DATABASE_URL_POOLED or DATABASE_URL_DIRECT in your environment.")
else:
    st.success("Database env var detected (connection will be implemented in the next steps).")