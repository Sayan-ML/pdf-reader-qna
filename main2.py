import os
import io
import time
import smtplib
from email.message import EmailMessage
from typing import List, Tuple, Dict
import re
import sqlite3
import tempfile

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.utilities import SerpAPIWrapper

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# --------------------------- DATABASE ---------------------------
def init_db():
    conn = sqlite3.connect("qa_history.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT,
            source TEXT
        )
    """)
    conn.commit()

    if "db_cleared" not in st.session_state:
        cursor.execute("DELETE FROM history")
        conn.commit()
        st.session_state["db_cleared"] = True

    conn.close()


def save_to_db(question, answer, source):
    conn = sqlite3.connect("qa_history.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO history (question, answer, source) VALUES (?, ?, ?)", (question, answer, source))
    conn.commit()
    conn.close()

def load_questions():
    conn = sqlite3.connect("qa_history.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, question FROM history")
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_answers_by_ids(ids):
    conn = sqlite3.connect("qa_history.db")
    cursor = conn.cursor()
    cursor.execute(f"SELECT question, answer FROM history WHERE id IN ({','.join('?'*len(ids))})", ids)
    rows = cursor.fetchall()
    conn.close()
    return rows

# --------------------------- PDF GENERATION ---------------------------
def generate_pdf_file(question, answer, filename="response.pdf"):
    """(Kept for sidebar combined-PDF flow that prefers file paths.)"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"<b>Question:</b> {question}", styles["Normal"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Answer:</b> {answer}", styles["Normal"]))

    doc.build(story)
    buffer.seek(0)

    with open(filename, "wb") as f:
        f.write(buffer.read())
    return filename

def generate_pdf_bytes(question, answer) -> bytes:
    """Return PDF as bytes for download/email (no temp file needed)."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"<b>Question:</b> {question}", styles["Normal"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Answer:</b> {answer}", styles["Normal"]))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def generate_combined_pdf(qas, filename="combined_answers.pdf"):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    for q, a in qas:
        story.append(Paragraph(f"<b>Question:</b> {q}", styles["Normal"]))
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"<b>Answer:</b> {a}", styles["Normal"]))
        story.append(Spacer(1, 18))
        story.append(Paragraph("<hr/>", styles["Normal"]))
        story.append(Spacer(1, 18))

    doc.build(story)
    buffer.seek(0)

    with open(filename, "wb") as f:
        f.write(buffer.read())
    return filename

# --------------------------- UI SETUP ---------------------------
st.set_page_config(
    page_title="PDF QA + Web Search + Email",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------- CUSTOM THEME ---------------------------
# --------------------------- CUSTOM THEME ---------------------------
st.markdown(
    """
    <style>
        /* App background */
        .stApp {
            background: linear-gradient(120deg, #fdfbfb 0%, #ebedee 100%);
            font-family: "Segoe UI", sans-serif;
        }

        /* Sidebar container */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%) !important;
            color: white !important;
        }

        /* Sidebar headers */
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4,
        section[data-testid="stSidebar"] label {
            color: #ecf0f1 !important;
        }

        /* Text inputs and number inputs */
        .stTextInput input, .stNumberInput input, .stPasswordInput input {
            background-color: #f4f6fa !important;
            border: 1px solid #d0d7e2 !important;
            border-radius: 10px !important;
            color: #2c3e50 !important;
            padding: 8px 12px !important;
        }
        .stTextInput input:focus, .stNumberInput input:focus, .stPasswordInput input:focus {
            border: 1px solid #4facfe !important;
            outline: none !important;
            box-shadow: 0 0 4px rgba(79, 172, 254, 0.6);
        }

        /* Dropdowns (selectbox) */
        .stSelectbox [data-baseweb="select"] {
            background-color: #f4f6fa !important;
            border-radius: 10px !important;
            color: #2c3e50 !important;
        }

        /* Sliders */
        .stSlider > div[data-baseweb="slider"] {
            background: transparent !important;
        }
        .stSlider [role="slider"] {
            background-color: #4facfe !important;
            border: 2px solid white !important;
        }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            border-radius: 12px;
            border: none;
            font-weight: 600;
            padding: 0.6em 1.2em;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            transform: scale(1.05);
            background: linear-gradient(90deg, #43e97b 0%, #38f9d7 100%);
            color: white;
        }
        /* Ensure text is visible on light backgrounds */
        .stApp, .stMarkdown, .stTextInput, .stNumberInput, .stPasswordInput, .stSelectbox, .stSlider {
            color: #2c3e50 !important;  /* dark text */
        }
        
        /* Make sure paragraph and headers are dark in main area */
        .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
            color: #2c3e50 !important;
        }
        
        /* Keep sidebar text white */
        section[data-testid="stSidebar"] {
            color: white !important;
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span {
            color: white !important;
        }
        
        /* Input text inside sidebar stays dark */
        section[data-testid="stSidebar"] input {
            color: #2c3e50 !important;
            background-color: #f4f6fa !important;
        }

    </style>
    """,
    unsafe_allow_html=True
)


st.title("📖 Smart PDF Q&A Assistant with Web Search & Email")

with st.sidebar:
    st.header("🔧 Settings")
    google_key = st.text_input("GOOGLE_API_KEY", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
    llm_model = st.selectbox(
        "Gemini Model",
        ["gemini-1.5-pro", "gemini-2.5-pro", "gemini-2.5-flash"],
        index=0
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    serpapi_key = st.text_input("SerpAPI API Key", type="password", value=os.getenv("SERPAPI_API_KEY", ""))

    st.markdown("---")
    st.subheader("📧 Email Settings (optional)")
    st.caption("You can enter SMTP username & password or leave them blank to use default credentials. Only recipient email is required.")
    smtp_host = st.text_input("SMTP host", value=os.getenv("SMTP_HOST", "smtp.gmail.com"))
    smtp_port = st.number_input("SMTP port", value=int(os.getenv("SMTP_PORT", 587)), step=1)
    smtp_user = st.text_input("SMTP email", value=os.getenv("SMTP_USER", "")) or os.getenv("DEFAULT_SMTP_USER", "sayan.banerjee1221@gmail.com")
    smtp_pass = st.text_input("SMTP password/app password", type="password", value=os.getenv("SMTP_PASS", "")) or os.getenv("DEFAULT_SMTP_PASS", "goweydofhmjuppjj")
    recipient_email = st.text_input("Recipient email (send refe rences)")

    st.markdown("---")
    top_k = st.slider("Retriever k", 1, 10, 4)
    score_threshold = st.slider("Score threshold (0-1)", 0.0, 1.0, 0.5, 0.05)

# --------------------------- HELPERS ---------------------------
def build_vectorstore_from_pdfs(uploaded_files):
    all_docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_path = tmp.name

        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        all_docs.extend(docs)
        os.remove(temp_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(all_docs)

    eembeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, eembeddings)

    return vectorstore, texts

def make_llm(google_key: str, model: str, temperature: float):
    if not google_key:
        st.warning("Please provide a GOOGLE_API_KEY to run the Gemini LLM.")
    return ChatGoogleGenerativeAI(google_api_key=google_key, model=model, temperature=temperature)

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Answer strictly using the provided context from the PDF. "
     "If the answer is not fully supported by the context, say 'INSUFFICIENT_CONTEXT'. "
     "Cite sources using [filename pagenum] style when available from metadata."),
    ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer succinctly.")
])

WEB_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful research assistant. Using the web search results below, answer the user's question. "
     "Include a brief list of the most relevant references as markdown bullet links with titles and URLs."),
    ("human", "Question: {question}\n\nSearch Results:\n{web_results}\n\nAnswer succinctly.")
])

def format_context(docs: List[Tuple[Document, float]]) -> str:
    lines = []
    for d, score in docs:
        src = d.metadata.get("source_file", d.metadata.get("source", "unknown"))
        page = d.metadata.get("page", None)
        tag = f"[{src}{' p'+str(page+1) if page is not None else ''}]"
        lines.append(f"(score={score:.2f}) {tag}\n{d.page_content}")
    return "\n\n---\n\n".join(lines)

def run_rag(llm: ChatGoogleGenerativeAI, vectordb: FAISS, question: str, top_k: int, score_threshold: float) -> Tuple[str, List[Tuple[Document, float]]]:
    docs_scores = vectordb.similarity_search_with_score(question, k=top_k)
    filtered = [(doc, score) for doc, score in docs_scores if score is not None]
    context = format_context(filtered)
    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"question": question, "context": context})
    return answer, filtered

def run_web_search_and_answer(llm: ChatGoogleGenerativeAI, question: str, max_results: int = 5, serpapi_key: str = None) -> Tuple[str, List[Dict[str, str]]]:
    if not serpapi_key:
        return "No SerpAPI key provided. Please enter one in the sidebar.", []

    try:
        tool = SerpAPIWrapper(serpapi_api_key=serpapi_key)
        search_results = tool.run(question)  # returns a string summary
    except Exception as e:
        return f"Web search failed: {e}", []

    if not search_results.strip():
        return "No relevant search results found.", []

    chain = WEB_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"question": question, "web_results": search_results})

    refs = []
    urls = re.findall(r"(https?://\S+)", search_results)
    for u in urls[:max_results]:
        refs.append({"title": u.split("/")[2], "url": u})

    return answer, refs

def send_email(
    smtp_host,
    smtp_port,
    smtp_user,
    smtp_pass,
    to_addr,
    subject,
    body,
    attachment_path=None,
    attachment_bytes: bytes = None,
    attachment_filename: str = "response.pdf",
):
    msg = EmailMessage()
    msg["From"] = smtp_user
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body)

    if attachment_bytes is not None:
        msg.add_attachment(
            attachment_bytes,
            maintype="application",
            subtype="pdf",
            filename=attachment_filename
        )
    elif attachment_path:
        with open(attachment_path, "rb") as f:
            msg.add_attachment(
                f.read(),
                maintype="application",
                subtype="pdf",
                filename=os.path.basename(attachment_path)
            )

    if int(smtp_port) == 465:
        server = smtplib.SMTP_SSL(smtp_host, int(smtp_port))
    else:
        server = smtplib.SMTP(smtp_host, int(smtp_port))
        server.starttls()

    try:
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)
    finally:
        server.quit()

# --------------------------- MAIN APP ---------------------------
init_db()

# Session state for current result (persists across reruns)
if "current_result" not in st.session_state:
    st.session_state["current_result"] = None

st.markdown("Upload one or more PDFs. Ask questions. If the answer isn't in the PDFs, I'll search the web and you can email yourself the references.")

uploaded = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True)
user_question = st.text_input("Your question")

col_a, col_b = st.columns([1, 1])
with col_a:
    build_btn = st.button("Build/Refresh Index", type="primary", key="build_index")
with col_b:
    ask_btn = st.button("Ask", key="ask_question")

# Init vector store holders
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
    st.session_state.chunks = []

# Build vectorstore
if build_btn:
    if not uploaded:
        st.warning("Please upload at least one PDF.")
    else:
        with st.spinner("Indexing PDFs..."):
            vectordb, chunks = build_vectorstore_from_pdfs(uploaded)
        st.session_state.vectordb = vectordb
        st.session_state.chunks = chunks
        st.success(f"Indexed {len(chunks)} chunks from {len(uploaded)} PDF(s).")

# Ask flow (stores result into session_state)
if ask_btn:
    if not user_question.strip():
        st.warning("Please type a question.")
    else:
        llm = make_llm(google_key, llm_model, temperature)
        if not llm or not google_key:
            st.stop()

        rag_answer, rag_docs = None, []
        used_web = False
        web_refs: List[Dict[str, str]] = []

        if st.session_state.vectordb is not None:
            with st.spinner("Answering from PDFs..."):
                answer, docs = run_rag(llm, st.session_state.vectordb, user_question, top_k, score_threshold)
                rag_answer, rag_docs = answer, docs

        needs_web = (rag_answer is None) or ("INSUFFICIENT_CONTEXT" in rag_answer.upper()) or (len(rag_docs) == 0)

        if needs_web:
            used_web = True
            with st.spinner("Searching the web..."):
                web_answer, web_refs = run_web_search_and_answer(llm, user_question, serpapi_key=serpapi_key)
            clean_answer = re.sub(r"\[.*?\]", "", web_answer or "").strip()
            source = "web"
        else:
            clean_answer = re.sub(r"\[.*?\]", "", rag_answer or "").strip()
            source = "pdf"

        # Generate PDF bytes and persist result for later buttons
        pdf_filename = "web_response.pdf" if source == "web" else "response.pdf"
        pdf_bytes = generate_pdf_bytes(user_question, clean_answer)

        # Save to DB
        save_to_db(user_question, clean_answer, source)

        # Persist in session_state for stable UI across reruns
        st.session_state["current_result"] = {
            "question": user_question,
            "answer": clean_answer,
            "source": source,
            "references": web_refs if source == "web" else [],
            "pdf_bytes": pdf_bytes,
            "pdf_filename": pdf_filename,
        }

# Show the latest result (if any), with stable Download + Email controls
current = st.session_state.get("current_result")
if current:
    st.subheader("Latest Answer")
    st.write(current["answer"])

    if current["source"] == "web" and current["references"]:
        st.markdown("**References:**")
        for r in current["references"]:
            st.markdown(f"- [{r['title']}]({r['url']})")

    # Download current PDF
    st.download_button(
        "⬇️ Download Answer as PDF",
        data=current["pdf_bytes"],
        file_name=current["pdf_filename"],
        mime="application/pdf",
        key="download_current_pdf"
    )

    # Email current result (separate, stable button)
    can_email = all([recipient_email, smtp_user, smtp_pass, smtp_host, smtp_port])
    if can_email:
        if st.button("📧 Send Current Answer via Email", key="send_current_email"):
            try:
                # Build body with references only for web
                body_lines = [
                    "Here is the answer to your question.",
                    f"\nQ: {current['question']}",
                    f"\nA: {current['answer']}\n",
                ]
                if current["source"] == "web" and current["references"]:
                    body_lines.append("References:\n" + "\n".join(f"- {r['title']}: {r['url']}" for r in current["references"]))
                body_text = "\n".join(body_lines)

                send_email(
                    smtp_host, int(smtp_port), smtp_user, smtp_pass,
                    recipient_email,
                    subject="Your Answer",
                    body=body_text,
                    attachment_bytes=current["pdf_bytes"],
                    attachment_filename=current["pdf_filename"],
                )
                st.success(f"📧 Email sent to {recipient_email} with the answer PDF attached!")
            except Exception as e:
                st.error(f"❌ Failed to send email: {e}")
    else:
        st.info("Fill out email settings in the sidebar to enable sending.")

# --------------------------- SIDEBAR HISTORY ---------------------------
st.sidebar.subheader("📖 Past Questions")
questions = load_questions()

if questions:
    q_options = {str(q[0]): q[1] for q in questions}
    selected_ids = st.sidebar.multiselect(
        "Select questions to include:",
        list(q_options.keys()),
        format_func=lambda x: q_options[x]
    )

    if selected_ids and st.sidebar.button("📧 Send Combined PDF", key="send_combined_pdf"):
        selected_qas = get_answers_by_ids(selected_ids)

        combined_pdf = generate_combined_pdf(selected_qas, filename="combined_answers.pdf")

        with open(combined_pdf, "rb") as f:
            st.download_button("⬇️ Download Combined PDF", f, file_name="combined_answers.pdf", key="download_combined_pdf")

        if all([recipient_email, smtp_user, smtp_pass, smtp_host, smtp_port]):
            try:
                # Build email body with Q&As
                body_lines = ["Here are the combined answers you selected:\n"]
                for q, a in selected_qas:
                    body_lines.append(f"Q: {q}\nA: {a}\n")
                body_text = "\n".join(body_lines)

                send_email(
                    smtp_host, int(smtp_port), smtp_user, smtp_pass,
                    recipient_email,
                    "Your Selected Answers",
                    body=body_text,   # ✅ now Q&As included
                    attachment_path=combined_pdf
                )
                st.success(f"📧 Combined answers sent to {recipient_email}!")
            except Exception as e:
                st.error(f"❌ Failed to send combined email: {e}")

else:
    st.sidebar.info("No Q&As saved yet.")

st.markdown("---")
st.caption("Built with LangChain, FAISS, Sqlite3 ,sentence-transformers, SerpAPI, Google Gemini API, and Streamlit.")
# --------------------------- ABOUT BUTTON ---------------------------
if st.button("ℹ️ About this App"):
    st.markdown("""
    ### 📚 What this App Does
    This app helps you:
    - Upload and read one or more **PDFs**
    - Ask questions about the PDF content
    - If the answer is not found, it will **search the web**
    - Save answers in a **database**
    - Select past Q&As from history and send them as a **combined PDF + text email**
    - Download answers as **PDF reports**
    - **Email answers** (single or combined) directly to yourself

    ---

    ### ⚙️ How it Works
    1. **Upload PDFs** → Click **Build/Refresh Index**  
    2. **Ask a Question** → Click **Ask**  
    3. The app first tries to answer from PDFs  
    4. If not found → falls back to **Web Search**  
    5. The answer is shown on screen and can be downloaded as PDF  
    6. Each Q&A is saved in the database for later reference  
    7. From the sidebar, select multiple Q&As → **Download Combined PDF** or **Email Combined Answers**  

    ---

    ### 📧 Email Setup
    To send answers via email, provide:
    - **SMTP host** (default: smtp.gmail.com)  
    - **SMTP port** (465 or 587)  
    - **SMTP email (username)**  
    - **SMTP password / App password**  
    - **Recipient email**  

    ⚠️ **Tip for Gmail users:**  
    - Go to [Google Account Security → App Passwords](https://myaccount.google.com/apppasswords)  
    - Generate a 16-character password  
    - Use that as your **SMTP password** here  

    ---

    ### 🔑 API Keys Required
    - **Google Gemini API Key** → Get it from [Google AI Studio](https://aistudio.google.com/app/apikey)  
    - **SerpAPI Key** → Create a free account at [SerpAPI](https://serpapi.com/)  

    Paste your keys into the sidebar inputs to activate features.

    ---

    ✅ Built with **LangChain, FAISS, Sqlite3, sentence-transformers, SerpAPI, Google Gemini API, and Streamlit**.  
    """)

st.markdown("Made by Sayan Banerjee | [GitHub](https://github.com/Sayan-ML)")


