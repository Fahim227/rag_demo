import os
import time
from pathlib import Path

import inngest
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="RAG PDF", page_icon="📄", layout="centered")

UPLOAD_DIR = Path("uploads")
INNGEST_API_BASE = os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=False)


def save_uploaded_pdf(file) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOAD_DIR / file.name
    file_path.write_bytes(file.getbuffer())
    return file_path


def send_event(name: str, data: dict) -> str:
    ids = get_inngest_client().send_sync(inngest.Event(name=name, data=data))
    return ids[0]


def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 0.5) -> dict:
    deadline = time.time() + timeout_s
    last_status = None
    while time.time() < deadline:
        resp = requests.get(f"{INNGEST_API_BASE}/events/{event_id}/runs", timeout=10)
        resp.raise_for_status()
        runs = resp.json().get("data", [])
        if runs:
            run = runs[0]
            last_status = run.get("status")
            if last_status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if last_status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Function run {last_status}")
        time.sleep(poll_interval_s)
    raise TimeoutError(f"Timed out waiting for run output (last status: {last_status})")


st.title("Upload a PDF to ingest")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    with st.spinner("Uploading and triggering ingestion..."):
        path = save_uploaded_pdf(uploaded)
        send_event("rag/ingest_pdf", {"pdf_path": str(path.resolve()), "source_id": path.name})
    st.success(f"Triggered ingestion for: {path.name}")
    st.caption("You can upload another PDF if you like.")

st.divider()
st.title("Ask a question about your PDFs")

with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input("How many chunks to retrieve", min_value=1, max_value=20, value=5, step=1)
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    with st.spinner("Sending event and generating answer..."):
        event_id = send_event("rag/query_pdf_ai", {"question": question.strip(), "top_k": int(top_k)})
        try:
            output = wait_for_run_output(event_id)
        except (RuntimeError, TimeoutError, requests.RequestException) as e:
            st.error(str(e))
            output = None

    if output:
        st.subheader("Answer")
        st.write(output.get("answer", "") or "(No answer)")
        sources = output.get("sources", [])
        if sources:
            st.caption("Sources")
            for s in sources:
                st.write(f"- {s}")
