"""
app.py
======
Enhanced Streamlit Web Application:
"Personal Knowledge Base — RAG Question Answering System"
College Laboratory Experiment 8 for B.Tech CSE (AI & ML).

Features:
- Hybrid Retrieval (Semantic FAISS + Lexical Matching).
- Query Intent Classification (Identity, Education, Internships, Skills, Projects, Publications).
- Multi-chunk aggregation for list queries (all internships/skills retrieved).
- Comprehensive Document Extraction Preview (page counts, character counts, scanned PDF detection).
- RAG Diagnostics Panel for live pipeline debugging.
- One-click "Rebuild Knowledge Base" with stale data prevention.
"""

import os
import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from document_processor import DocumentProcessor
from rag_pipeline import RAGPipeline

# Directories
DOCS_DIR = BASE_DIR / "Student Knowledge Base"
VECTOR_DIR = BASE_DIR / "vector_store"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"

DOCS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="Personal Knowledge Base — RAG QA System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# Custom Styling (Academic Laboratory Theme)
# ----------------------------------------------------
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        color: #f8fafc;
        margin-bottom: 12px;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-success { background-color: #10b981; color: #ffffff; }
    .badge-warning { background-color: #f59e0b; color: #ffffff; }
    .badge-info { background-color: #3b82f6; color: #ffffff; }
    .badge-error { background-color: #ef4444; color: #ffffff; }
    .badge-purple { background-color: #8b5cf6; color: #ffffff; }
    .badge-gray { background-color: #475569; color: #f1f5f9; }
    
    .source-tag {
        background-color: #2563eb;
        color: white;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 500;
        display: inline-block;
        margin: 2px 6px 2px 0;
    }
    .chunk-box {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 14px;
        border-radius: 6px;
        margin-bottom: 10px;
        font-size: 0.92rem;
        color: #1e293b;
    }
    .unavailable-box {
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 16px;
        border-radius: 6px;
        color: #991b1b;
        font-weight: 500;
        margin: 14px 0;
    }
    .answer-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 22px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-top: 15px;
    }
    .diag-box {
        background-color: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# ----------------------------------------------------
# Knowledge Base & Vector Store Management
# ----------------------------------------------------
def clear_knowledge_base_and_vector_store():
    """
    Clears all documents from the Student Knowledge Base folder (except readme/instruction files),
    wipes the FAISS vector index and metadata from disk and memory,
    resets preview widgets, and empties all session state.
    """
    # 1. Clear previous documents from DOCS_DIR
    if DOCS_DIR.exists():
        for file_path in DOCS_DIR.iterdir():
            if file_path.is_file() and file_path.name.lower() not in {"readme.txt", "readme.md"}:
                try:
                    file_path.unlink()
                except Exception:
                    pass

    # 2. Clear vector store files from disk (in both local and workspace root vector_store)
    for v_dir in [VECTOR_DIR, BASE_DIR.parent / "vector_store"]:
        if v_dir.exists():
            for f in v_dir.glob("*"):
                if f.is_file():
                    try:
                        f.unlink()
                    except Exception:
                        pass

    # 3. Clear pipeline in memory
    if "rag_pipeline" in st.session_state and st.session_state.rag_pipeline is not None:
        st.session_state.rag_pipeline.clear_index()

    # 4. Clear all preview widget keys and QA states from session state
    for k in list(st.session_state.keys()):
        if k.startswith("preview_") or k == "query_input":
            del st.session_state[k]

    st.session_state.indexed_files = []
    st.session_state.qa_history = []
    st.session_state.last_result = None
    st.session_state.selected_question = ""


# ----------------------------------------------------
# Session State Initialization
# ----------------------------------------------------
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = RAGPipeline(
        vector_store_dir=str(VECTOR_DIR),
        similarity_threshold=0.20,
        top_k=6,
        candidate_count=14
    )

if "doc_processor" not in st.session_state:
    st.session_state.doc_processor = DocumentProcessor(chunk_size=650, chunk_overlap=120)

# Whenever the code is run initially / fresh session:
# Clear the vector base and knowledge base so it is empty every time it runs.
if "app_initialized" not in st.session_state:
    st.session_state.app_initialized = True
    clear_knowledge_base_and_vector_store()

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

if "selected_question" not in st.session_state:
    st.session_state.selected_question = ""

if "last_result" not in st.session_state:
    st.session_state.last_result = None

pipeline: RAGPipeline = st.session_state.rag_pipeline
processor: DocumentProcessor = st.session_state.doc_processor

# ----------------------------------------------------
# Core Indexing Actions
# ----------------------------------------------------
def rebuild_knowledge_base():
    """
    Completely rescans the Student Knowledge Base folder,
    extracts all files, produces previews, and rebuilds the FAISS index.
    """
    chunks, successful_reports, failed_reports = processor.process_directory(str(DOCS_DIR))
    file_hashes = {r["filename"]: r["file_hash"] for r in successful_reports}
    
    if chunks:
        pipeline.build_index(chunks, file_hashes=file_hashes)
        st.session_state.indexed_files = sorted([r["filename"] for r in successful_reports])
    else:
        pipeline.clear_index()
        st.session_state.indexed_files = []

    total_chars = sum(r.get("char_count", 0) for r in successful_reports)
    return len(chunks), successful_reports, failed_reports, total_chars


def process_and_index_uploads(uploaded_files):
    """Saves uploaded files to disk and triggers a clean rebuild."""
    for uf in uploaded_files:
        dest = DOCS_DIR / uf.name
        with open(dest, "wb") as f:
            f.write(uf.read())
    return rebuild_knowledge_base()


# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------
with st.sidebar:
    st.title("🎓 Lab Experiment 8")
    st.markdown("**Personal RAG Question Answering**")
    st.caption("B.Tech CSE (AI & ML) — Academic Laboratory")
    st.divider()

    st.subheader("📄 Supported Formats")
    st.markdown("""
    <span class="badge badge-info">PDF</span>
    <span class="badge badge-info">DOCX</span>
    <span class="badge badge-info">TXT</span>
    <span class="badge badge-info">MD</span>
    """, unsafe_allow_html=True)
    st.divider()

    # Ollama LLM Status
    st.subheader("🤖 Local LLM (Ollama)")
    ollama_info = pipeline.check_ollama_status()
    if ollama_info["available"]:
        st.markdown('<span class="badge badge-success">● Ollama Connected</span>', unsafe_allow_html=True)
        available_models = ollama_info["models"]
        if available_models:
            preferred_idx = 0
            for idx, m in enumerate(available_models):
                if "llama" in m.lower() or "qwen" in m.lower():
                    preferred_idx = idx
                    break
            selected_model = st.selectbox(
                "Select Model",
                options=available_models,
                index=preferred_idx,
                help="Local model running via Ollama"
            )
            pipeline.ollama_model = selected_model
        else:
            st.warning("Ollama is running, but no models found. Run `ollama pull llama3:latest`.")
    else:
        st.markdown('<span class="badge badge-error">● Ollama Offline</span>', unsafe_allow_html=True)
        st.info("System will safely operate in **Extractive Fallback Mode** (retrieval & citations active).")

    st.divider()

    # Index Statistics
    st.subheader("📊 Knowledge Base Stats")
    num_docs = len(st.session_state.indexed_files)
    num_chunks = len(pipeline.chunks_metadata)

    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        st.metric("Documents", num_docs)
    with col_sb2:
        st.metric("Chunks", num_chunks)

    st.divider()

    # Retrieval Configuration
    st.subheader("⚙️ Retrieval Parameters")
    sim_threshold = st.slider(
        "Similarity Threshold (Cosine)",
        min_value=0.05,
        max_value=0.70,
        value=pipeline.similarity_threshold,
        step=0.05,
        help="Chunks with similarity below this threshold are rejected to avoid hallucination."
    )
    pipeline.similarity_threshold = sim_threshold

    top_k_chunks = st.slider(
        "Top-K Chunks to Retrieve",
        min_value=2,
        max_value=12,
        value=pipeline.top_k,
        step=1,
        help="Target number of candidate passages passed to the LLM context."
    )
    pipeline.top_k = top_k_chunks

    candidate_count_val = st.slider(
        "Candidates Evaluated (Hybrid)",
        min_value=6,
        max_value=24,
        value=pipeline.candidate_count,
        step=2,
        help="Number of initial dense candidates retrieved for lexical re-ranking."
    )
    pipeline.candidate_count = candidate_count_val


# ----------------------------------------------------
# MAIN DASHBOARD AREA
# ----------------------------------------------------
st.title("Personal Knowledge Base — RAG Question Answering System")
st.markdown(
    "Query your personal academic and career records with zero hallucinations. "
    "All responses are grounded strictly in your personal documents."
)

tab_qa, tab_docs, tab_diag, tab_arch = st.tabs([
    "💬 Question Answering",
    "📁 Document Management & Preview",
    "🩺 RAG Diagnostics",
    "🔬 System Architecture & Theory"
])

# ====================================================
# TAB 1: QUESTION ANSWERING
# ====================================================
with tab_qa:
    st.subheader("Ask a Question About Your Records")

    # Sample Question Buttons
    st.markdown("**Quick Sample Questions:**")
    sample_questions = [
        "What is my full name?",
        "Which college or university do I attend?",
        "Which internships have I undertaken?",
        "What are the student's research interests?",
        "What technical skills do I have?",
        "What projects has the student completed?",
        "What is my CGPA or percentage?",
        "What research publications has the student authored?"
    ]

    cols_sq1, cols_sq2 = st.columns(2)
    for idx, q_text in enumerate(sample_questions):
        target_col = cols_sq1 if idx % 2 == 0 else cols_sq2
        if target_col.button(f"👉 {q_text}", key=f"sq_{idx}", use_container_width=True):
            st.session_state.selected_question = q_text

    # Question Input
    default_q = st.session_state.selected_question or ""
    user_query = st.text_input(
        "Enter your question:",
        value=default_q,
        placeholder="e.g., Which internships have I undertaken?",
        key="query_input"
    )

    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        ask_clicked = st.button("🔍 Ask Question", type="primary", use_container_width=True)
    with col_btn2:
        if st.button("🧹 Clear Question", use_container_width=False):
            st.session_state.selected_question = ""
            st.rerun()

    if ask_clicked:
        if not user_query.strip():
            st.warning("Please type a question or select one of the sample questions above.")
        elif num_chunks == 0:
            st.error("⚠️ The knowledge base is currently empty! Please upload or index your documents in the 'Document Management' tab first.")
        else:
            with st.spinner("Retrieving relevant context via hybrid search and generating grounded answer..."):
                result = pipeline.answer_question(
                    query=user_query,
                    top_k=pipeline.top_k,
                    similarity_threshold=pipeline.similarity_threshold
                )

                st.session_state.qa_history.insert(0, {
                    "question": user_query,
                    "result": result
                })
                st.session_state.last_result = result

    # Display Last Result
    if st.session_state.last_result:
        res = st.session_state.last_result
        st.markdown("---")
        st.subheader("💡 Answer")

        answer_text = res.get("answer", "")
        status = res.get("status", "")
        q_type = res.get("query_type", "GENERAL")
        sources = res.get("sources", [])
        retrieved_chunks = res.get("retrieved_chunks", [])
        llm_used = res.get("llm_used", "None")

        # Category Badge
        st.markdown(f'<span class="badge badge-purple">Category: {q_type}</span>', unsafe_allow_html=True)

        if status in {"no_relevant_context", "missing_fact"}:
            st.markdown(f"""
            <div class="unavailable-box">
                <strong>🚫 Information Unavailable:</strong><br>
                {answer_text}
            </div>
            """, unsafe_allow_html=True)
            st.info(f"💡 The system verified that this detail is not established by the indexed documents and refused to hallucinate.")
        elif status == "fallback":
            st.markdown('<span class="badge badge-warning">Extractive Fallback Mode</span>', unsafe_allow_html=True)
            st.markdown(answer_text)
        else:
            st.markdown(f'<span class="badge badge-success">Grounded by {llm_used}</span>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="answer-card">
                {answer_text}
            </div>
            """, unsafe_allow_html=True)

        # Source Citations
        if sources:
            st.markdown("#### 📚 Source Document Citations")
            sources_html = "".join([f'<span class="source-tag">📄 {s}</span>' for s in sources])
            st.markdown(sources_html, unsafe_allow_html=True)

        # Expandable Retrieved Passages
        if retrieved_chunks:
            with st.expander(f"🔍 Inspect {len(retrieved_chunks)} Retrieved Context Passage(s) with Similarity & Lexical Scores", expanded=False):
                for idx, ch in enumerate(retrieved_chunks, 1):
                    dense_score = ch.get("similarity_score", 0.0)
                    lex_score = ch.get("lexical_score", 0.0)
                    hybrid_score = ch.get("hybrid_score", 0.0)
                    sec = ch.get("section_hint", "GENERAL")
                    st.markdown(f"""
                    <div class="chunk-box">
                        <strong>Passage #{idx}</strong> — File: <code>{ch.get('source')}</code> | 
                        Chunk ID: <code>{ch.get('chunk_id')}</code> | Section: <code>{sec}</code><br>
                        Dense Score: <strong>{dense_score:.4f}</strong> | 
                        Lexical Score: <strong>{lex_score:.4f}</strong> | 
                        Hybrid Score: <strong>{hybrid_score:.4f}</strong>
                        <hr style="margin: 6px 0; border: none; border-top: 1px dashed #cbd5e1;">
                        {ch.get('text')}
                    </div>
                    """, unsafe_allow_html=True)

    # QA History
    if len(st.session_state.qa_history) > 1:
        with st.expander("🕒 Past Queries in this Session", expanded=False):
            for item in st.session_state.qa_history[1:]:
                st.markdown(f"**Q:** {item['question']}")
                st.caption(f"Category: {item['result'].get('query_type')} | Status: {item['result'].get('status')}")
                st.markdown(f"> {item['result'].get('answer')[:200]}...")
                st.divider()


# ====================================================
# TAB 2: DOCUMENT MANAGEMENT & PREVIEW
# ====================================================
with tab_docs:
    st.subheader("Document Management & Text Extraction Preview")
    st.markdown(
        "Inspect the raw extracted text of your documents before indexing. "
        "Verify that names, educational institutions, skills, and all internships are clearly extracted."
    )

    col_dm1, col_dm2 = st.columns([3, 2])

    with col_dm1:
        st.markdown("#### 📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files to add to your knowledge base (.pdf, .docx, .txt, .md)",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True
        )

        if st.button("⚡ Process & Index Uploaded Files", type="primary", disabled=not uploaded_files):
            with st.spinner("Processing uploads, cleaning text, and building FAISS vector index..."):
                num_chunks, succ, fail, total_chars = process_and_index_uploads(uploaded_files)
                if succ:
                    st.success(f"Indexed {len(succ)} file(s), {total_chars} characters into {num_chunks} chunks!")
                if fail:
                    for err in fail:
                        st.error(f"❌ {err['filename']}: {err['error']}")
                st.rerun()

    with col_dm2:
        st.markdown("#### 🔄 Rebuild Knowledge Base")
        st.caption(f"Folder: `{DOCS_DIR}`")
        if st.button("🔨 Rebuild Knowledge Base", use_container_width=True, type="secondary"):
            with st.spinner("Scanning directory, extracting text, and generating new FAISS index..."):
                num_chunks, succ, fail, total_chars = rebuild_knowledge_base()
                if succ:
                    st.success(f"Rebuilt index with {len(succ)} file(s), {total_chars} total characters, and {num_chunks} chunks.")
                else:
                    st.warning("No valid documents found in Student Knowledge Base folder.")
                if fail:
                    for err in fail:
                        st.error(f"❌ {err['filename']}: {err['error']}")
                st.rerun()

        if st.button("🗑️ Clear Vector Database", use_container_width=True):
            clear_knowledge_base_and_vector_store()
            st.success("Vector database and knowledge base cleared successfully. Previous documents removed.")
            st.rerun()

    st.divider()

    # Document Extraction Inspection Section
    st.subheader("🔍 Document Extraction Inspection & Preview")
    st.caption("Inspect extracted text before indexing to detect empty or scanned image documents.")

    local_files = [f for f in sorted(DOCS_DIR.iterdir()) if f.is_file() and f.name.lower() not in {"readme.txt", "readme.md"}]

    if local_files:
        for file_path in local_files:
            rep = processor.inspect_and_extract_file(str(file_path))
            fname = rep["filename"]
            doc_ext = rep["doc_type"]
            char_count = rep["char_count"]
            word_count = rep["word_count"]
            num_pages = rep["num_pages"]

            with st.container():
                st.markdown(f"### 📄 `{fname}`")
                cols_rep = st.columns(4)
                cols_rep[0].markdown(f"**Format:** `{doc_ext}`")
                cols_rep[1].markdown(f"**Pages:** `{num_pages}`" if doc_ext == ".pdf" else "**Pages:** `1`")
                cols_rep[2].markdown(f"**Characters:** `{char_count:,}`")
                cols_rep[3].markdown(f"**Words:** `{word_count:,}`")

                if rep["is_scanned"]:
                    st.warning(f"⚠️ Scanned PDF Alert: '{fname}' has little or no text layer. OCR may be required to extract text.")
                elif rep["is_empty"]:
                    st.error(f"❌ Empty Document Alert: No readable text could be extracted from '{fname}'.")
                else:
                    st.success(f"✅ Text extraction valid ({char_count:,} characters preserved).")

                # Expandable Raw Text Preview
                with st.expander(f"👁️ View Extracted Text for `{fname}`", expanded=False):
                    st.caption("⚠️ Notice: This preview displays raw extracted text and may contain personal information.")
                    st.text_area(
                        label=f"Extracted Text: {fname}",
                        value=rep["text"],
                        height=220,
                        key=f"preview_{fname}"
                    )
            st.divider()
    else:
        st.info("📭 Knowledge base is empty. No documents currently loaded for inspection. Upload documents above to view extraction previews and index them.")


# ====================================================
# TAB 3: RAG DIAGNOSTICS PANEL
# ====================================================
with tab_diag:
    st.subheader("🩺 RAG System Diagnostics")
    st.markdown("Live diagnostic inspection of extraction health, vector storage, and retrieval metrics.")

    col_dg1, col_dg2 = st.columns(2)

    with col_dg1:
        st.markdown("#### 📂 Ingestion & Vector Storage Health")
        st.markdown(f"""
        - **Total Indexed Documents:** `{len(st.session_state.indexed_files)}`
        - **Total Vector Chunks:** `{len(pipeline.chunks_metadata)}`
        - **Vector Dimension:** `{pipeline.embedding_dimension}` (Model: `{pipeline.embedding_model_name}`)
        - **Vector Index Status:** `{"Loaded & Ready" if pipeline.index is not None else "Empty / Not Built"}`
        """)

        # Resume verification check
        resume_found = any("resume" in f.lower() for f in st.session_state.indexed_files)
        if resume_found:
            st.success("✅ Student Resume detected in indexed documents.")
        else:
            st.info("ℹ️ No file with 'resume' in its name detected. Please ensure your resume is indexed.")

    with col_dg2:
        st.markdown("#### 🤖 LLM & Server Health")
        st.markdown(f"""
        - **Ollama Host URL:** `{pipeline.ollama_url}`
        - **Ollama Reachable:** `{"Yes (Connected)" if ollama_info["available"] else "No (Offline - Fallback active)"}`
        - **Selected Model:** `{pipeline.ollama_model}`
        - **Installed Models:** `{', '.join(ollama_info.get('models', [])) or 'None'}`
        """)

    st.divider()

    st.markdown("#### 🔍 Last Query Diagnostics")
    if st.session_state.last_result:
        res = st.session_state.last_result
        st.markdown(f"""
        - **User Query:** `{res.get('query', user_query)}`
        - **Detected Intent:** `{res.get('query_type')}`
        - **Execution Status:** `{res.get('status')}`
        - **Retrieved Chunks Count:** `{len(res.get('retrieved_chunks', []))}`
        - **LLM Engine Used:** `{res.get('llm_used')}`
        """)

        if res.get("retrieved_chunks"):
            st.markdown("**Retrieved Chunks Breakdown:**")
            for idx, c in enumerate(res.get("retrieved_chunks", []), 1):
                st.markdown(
                    f"- **Passage #{idx}** ({c['source']}, Chunk #{c['chunk_id']}): "
                    f"Hybrid Score: `{c.get('hybrid_score', 0):.4f}` | "
                    f"Dense: `{c.get('similarity_score', 0):.4f}` | "
                    f"Lexical: `{c.get('lexical_score', 0):.4f}` | "
                    f"Section: `{c.get('section_hint', 'GENERAL')}`"
                )
    else:
        st.info("No queries have been submitted yet in this session. Ask a question to view retrieval diagnostics.")


# ====================================================
# TAB 4: SYSTEM ARCHITECTURE & LAB INFO
# ====================================================
with tab_arch:
    st.subheader("Laboratory Experiment Details & Architecture")
    st.markdown("""
    ### 🎯 Experiment 8: Personal Knowledge Base using RAG
    **Objective:** Construct a domain-specific question answering system over personal documents
    using Hybrid Retrieval-Augmented Generation that enforces strict evidence grounding and zero hallucination.

    ---
    ### ⚙️ Enhanced System Specifications:
    """)

    col_sp1, col_sp2, col_sp3 = st.columns(3)
    with col_sp1:
        st.markdown("""
        **Document Processing:**
        - **PDF:** `pypdf` (Page-by-page extraction & Scanned detection)
        - **DOCX:** `python-docx` (Paragraphs + Tables)
        - **Text / MD:** UTF-8 / latin-1 fallback
        - **Chunk Size:** ~650 characters
        - **Chunk Overlap:** 120 characters
        - **Boundary Strategy:** Paragraph / Sentence Aware
        """)
    with col_sp2:
        st.markdown(f"""
        **Hybrid Search & Vector DB:**
        - **Embeddings:** `{pipeline.embedding_model_name}` (384-dim)
        - **Vector Index:** `FAISS IndexFlatIP` (Normalized Cosine)
        - **Lexical Search:** Stemmed Term Coverage + Entity Boost
        - **Default Threshold:** `{pipeline.similarity_threshold}`
        - **Default Top-K:** `{pipeline.top_k}`
        """)
    with col_sp3:
        st.markdown(f"""
        **Grounded LLM Generation:**
        - **Engine:** Ollama (Local)
        - **Active Model:** `{pipeline.ollama_model}`
        - **Temperature:** 0.05 (Deterministic grounding)
        - **Multi-chunk Aggregator:** Enforces full list retrieval for internships & skills
        - **Safety Fallback:** Direct Extractive Retrieval
        """)

    st.divider()

    st.markdown("""
    ### 🔄 Enhanced RAG Workflow
    ```
    [Personal Documents] (PDF, DOCX, TXT, MD)
           │
           ▼
    [Extraction & Unicode Normalization]
           │
           ▼
    [Boundary-Aware Chunking + Section Tagging]
           │
           ▼
    [Dual Indexing: FAISS Dense Vectors + Lexical Term Index]
           │
    [User Query] ──► [Query Intent Classifier] 
                          │
                          ├─► [Dense Embedding Search (Top-N Candidates)]
                          ├─► [Lexical Keyword Matching (Root/Stem Aware)]
                          └─► [Intent-Specific Section Boosting]
                                  │
                                  ▼
                     [Hybrid Score Fusion (RRF / Weighted)]
                                  │
                  ┌───────────────┴───────────────┐
                  │                               │
       [Candidate Passes Criteria?]       [No Relevant Chunks]
                  │                               │
                 YES                              ▼
                  │                 ["I could not find this information
                  ▼                  in the provided knowledge base."]
       [Structured Evidence Extraction]
                  │
                  ▼
       [Strict Prompt with All Excerpts]
                  │
                  ▼
          [Local Ollama LLM]
                  │
                  ▼
     [Grounded Answer + Citations]
    ```
    """)
