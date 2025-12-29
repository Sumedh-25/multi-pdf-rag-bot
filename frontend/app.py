import requests
import streamlit as st
from datetime import datetime
from pathlib import Path

# ==============================
# CONFIG
# ==============================
API_URL = "http://127.0.0.1:8000"
LOGO_PATH = Path(r"C:\Users\Sumedh\Multi_Pdf_Rag_Bot\frontend\MPRB_1.png")

st.set_page_config(
    page_title="Multi-PDF RAG Bot",
    page_icon=str(LOGO_PATH),
    layout="wide"
)

# ==============================
# SESSION STATE
# ==============================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ==============================
# SIDEBAR
# ==============================
with st.sidebar:
    st.image(str(LOGO_PATH), width=150)
    st.markdown("## 📂 Document Controls")

    # ---------- Upload PDFs ----------
    st.markdown("### 📤 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        with st.spinner("📤 Uploading PDFs to backend..."):
            try:
                files = [
                    ("files", (f.name, f.getvalue(), "application/pdf"))
                    for f in uploaded_files
                ]
                response = requests.post(f"{API_URL}/upload-pdf", files=files, timeout=120)
                response.raise_for_status()
                uploaded_names = response.json().get("files", [])
                st.success("✅ PDFs uploaded successfully")
                for name in uploaded_names:
                    st.write(f"• {name}")
            except Exception as e:
                st.error(f"❌ Upload failed: {e}")

    st.markdown("---")

    # ---------- Delete PDFs ----------
    st.markdown("### 🗑 Delete PDFs")
    try:
        pdf_list = requests.get(f"{API_URL}/list-pdfs").json().get("pdfs", [])
        selected_to_delete = st.multiselect("Select PDFs to delete", pdf_list)
        if st.button("Delete Selected PDFs") and selected_to_delete:
            with st.spinner("🗑 Deleting PDFs..."):
                response = requests.post(f"{API_URL}/delete-pdf", json={"filenames": selected_to_delete})
                if response.status_code == 200:
                    st.success(f"Deleted: {', '.join(selected_to_delete)}")
                else:
                    st.error("Failed to delete PDFs")
    except Exception as e:
        st.warning(f"Could not load PDFs: {e}")

    st.markdown("---")

    # ---------- RAG Controls ----------
    st.markdown("### ⚙️ RAG Settings")
    top_k = st.slider("Chunks to retrieve (k)", 5, 30, 20)
    strict_mode = st.toggle("Strict Document Mode", value=True)
    academic_mode = st.radio("Answer Style", ["Academic", "Simple"], horizontal=True)
    show_debug = st.checkbox("🔍 Show retrieved context (Debug)")

    st.markdown("---")
    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared")

# ==============================
# MAIN UI
# ==============================
st.image(str(LOGO_PATH), width=200)
st.caption("Ask accurate questions from your uploaded academic PDFs")

question = st.text_input("Enter your question", placeholder="e.g. Explain the Problem Statement, Objectives, and Scope...")
ask_clicked = st.button("🚀 Ask")

# ==============================
# QUERY HANDLING
# ==============================
if ask_clicked and question.strip():
    with st.spinner("🔍 Retrieving documents and generating answer..."):
        try:
            payload = {
                "question": question,
                "top_k": top_k,
                "strict": strict_mode,
                "mode": academic_mode.lower(),
            }
            response = requests.post(f"{API_URL}/ask", json=payload, timeout=180)
            response.raise_for_status()
            answer = response.json().get("answer", "")
            st.session_state.chat_history.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "question": question,
                "answer": answer
            })
        except Exception as e:
            st.error(f"❌ API Error: {e}")

# ==============================
# CHAT HISTORY
# ==============================
st.markdown("---")
st.subheader("💬 Conversation")
for chat in reversed(st.session_state.chat_history):
    st.markdown(f"**🧑 Question ({chat['time']}):**")
    st.write(chat["question"])
    st.markdown("**🤖 Answer:**")
    st.write(chat["answer"])
    st.markdown("---")
