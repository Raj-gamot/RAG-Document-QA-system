import streamlit as st
import requests
import os

st.set_page_config(
    page_title="RAG Document Q&A System",
    layout="wide",
)

API_URL = os.getenv("FASTAPI_URL", "http://localhost:8000").rstrip("/")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

st.markdown(
    """
    <style>
    body { background-color: #0e1117; color: #e6e6e6; }
    .app-header {
        background: linear-gradient(90deg, #4f46e5, #9333ea);
        padding: 20px; border-radius: 12px; margin-bottom: 20px; color: white;
    }
    section[data-testid="stSidebar"] { background-color: #0b0f14; }
    .user-msg {
        background-color: #2563eb; padding: 12px; border-radius: 12px;
        color: white; margin-bottom: 8px; max-width: 80%;
    }
    .assistant-msg {
        background-color: #1f2937; padding: 12px; border-radius: 12px;
        margin-bottom: 8px; max-width: 80%; border: 1px solid #374151;
    }
    button[kind="primary"] { background-color: #4f46e5 !important; border-radius: 8px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <h1>📄 RAG Document Q&A System</h1>
        <p>Upload documents and ask intelligent questions using Retrieval-Augmented Generation.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("<h2>📤 Upload Documents</h2>", unsafe_allow_html=True)

    try:
        response = requests.get(f"{API_URL}/supported-formats", timeout=3)
        supported_formats = (
            response.json()["formats"] if response.status_code == 200 else ["pdf", "docx"]
        )
    except Exception:
        supported_formats = ["pdf", "docx"]

    uploaded_file = st.file_uploader("Choose a file", type=supported_formats)

    if uploaded_file and st.button("Process Document", type="primary"):
        with st.spinner("Processing document..."):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type or "application/octet-stream",
                )
            }
            try:
                response = requests.post(f"{API_URL}/upload", files=files, timeout=120)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Processed {result['filename']}")
                    st.info(f"Total chunks: {result['total_chunks']}")
                    if uploaded_file.name not in st.session_state.uploaded_files:
                        st.session_state.uploaded_files.append(uploaded_file.name)
                else:
                    try:
                        err = response.json()
                        st.error(err.get("detail", response.text))
                    except Exception:
                        st.error(response.text)
            except Exception as e:
                st.error(str(e))

    if st.session_state.uploaded_files:
        st.markdown("---")
        st.subheader("📚 Uploaded Files")
        for f in st.session_state.uploaded_files:
            st.write(f"✓ {f}")

        if st.button("Clear knowledge base"):
            try:
                response = requests.delete(f"{API_URL}/documents", timeout=10)
                if response.status_code == 200:
                    st.session_state.uploaded_files = []
                    st.session_state.messages = []
                    st.success("Knowledge base cleared")
                    st.rerun()
                else:
                    st.error("Failed to clear knowledge base")
            except Exception as e:
                st.error(str(e))

    st.markdown("---")
    st.subheader("🔌 API Status")
    st.caption(f"Backend: {API_URL}")
    try:
        if requests.get(f"{API_URL}/health", timeout=3).status_code == 200:
            st.success("Connected")
        else:
            st.error("API error")
    except Exception:
        st.error("Disconnected — start FastAPI with: uvicorn main:app --reload")

st.markdown("<h2>💬 Ask Questions</h2>", unsafe_allow_html=True)

for msg in st.session_state.messages:
    css_class = "user-msg" if msg["role"] == "user" else "assistant-msg"
    st.markdown(f"<div class='{css_class}'>{msg['content']}</div>", unsafe_allow_html=True)

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                f"{API_URL}/query",
                json={"question": prompt},
                timeout=60,
            )
            if response.status_code == 200:
                answer = response.json().get("answer", "No answer returned.")
            else:
                try:
                    err = response.json()
                    answer = err.get("detail", response.text)
                except Exception:
                    answer = response.text

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()
        except Exception as e:
            st.error(str(e))

if st.session_state.messages and st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()
