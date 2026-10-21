# Personal Knowledge Base — RAG Question Answering System

A robust, beginner-friendly, and strictly grounded Retrieval-Augmented Generation (RAG) system that allows you to query your personal academic documents (resumes, transcripts, project reports, certificates, and notes) with **zero hallucination**. If information cannot be found in the uploaded documents, the system refuses to invent facts and responds:

> *"I could not find this information in the provided knowledge base."*

---

## 🛠️ Technology Stack & Key Repair Highlights

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core application logic (tested on Python 3.12) |
| **Frontend UI** | Streamlit | Clean, 4-tab interactive web interface with live diagnostics |
| **Doc Processing** | `pypdf`, `python-docx`, built-in I/O | PDF (page-by-page), DOCX, TXT, and Markdown parsing with Unicode cleaning |
| **Chunking** | Boundary-Aware Chunker | 500–800 chars with 100–150 overlap at paragraph/sentence boundaries |
| **Embeddings** | `sentence-transformers` | `all-MiniLM-L6-v2` (384-dimensional dense vectors) |
| **Vector DB** | FAISS (`faiss-cpu`) | `IndexFlatIP` with L2-normalized vectors (Cosine Similarity) |
| **Hybrid Search** | Semantic + Lexical | Reciprocal Rank Fusion of dense FAISS and stemmed keyword lexical matching |
| **Intent Classifier**| Query-Specific Strategies | Specialized retrieval for Identity, Education, Internships, Skills, and Projects |
| **Local LLM** | Ollama | `llama3:latest` running locally with zero-hallucination prompting |
| **Safety Fallback** | Extractive Retrieval | Displays retrieved passages directly with scores if Ollama is offline |

---

## 📂 Project Structure

```
Student-RAG-Knowledge-Base/
│
├── app.py                     # Streamlit frontend & interactive dashboard
├── rag_pipeline.py            # Embeddings, FAISS indexing, similarity retrieval & LLM call
├── document_processor.py      # Multi-format parsing, text cleaning & boundary chunking
├── test_rag.py                # Automated test suite (10 test cases)
├── requirements.txt           # Project dependencies
├── README.md                  # Comprehensive documentation and run guide
├── lab_record.md              # Academic laboratory record with Mermaid diagram
├── .gitignore                 # Excludes confidential personal docs & vector stores
│
├── Student Knowledge Base/    # Personal documents directory (.pdf, .docx, .txt, .md)
│   └── README.txt             # Instructions for dropping personal documents
│
├── vector_store/              # FAISS index (index.faiss) & chunk metadata (metadata.pkl)
│
└── screenshots/               # Folder for lab report execution screenshots
```

---

## 🚀 Step-by-Step Setup Guide (Windows PowerShell)

Follow these exact commands in your Windows PowerShell terminal:

### 1. Open the Project Directory
```powershell
cd "c:\Users\Aum Tripathi\Documents\Aum_Tripathi_Experiment_8\Student-RAG-Knowledge-Base"
```

### 2. (Optional) Create a Python Virtual Environment
```powershell
python -m venv venv
```

### 3. Activate the Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```
*(If PowerShell displays an execution policy error, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

### 4. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 5. Verify Ollama Installation & Status
Ollama runs lightweight LLMs locally without requiring paid API keys or cloud accounts.
Check if Ollama is installed:
```powershell
ollama --version
```
To check which models are already installed on your machine:
```powershell
ollama list
```
*(If you already have `llama3:latest` installed, the application will automatically detect and use it!)*

### 6. (Optional) Pull the Recommended Lightweight Model
To download `qwen2.5:3b` (fast 3B model, ~1.9 GB):
```powershell
ollama pull qwen2.5:3b
```

### 7. Run Automated Tests
Before starting the user interface, verify that document parsing, FAISS indexing, and fallback mechanisms work properly:
```powershell
python test_rag.py
```
*(All 10 tests should pass with `OK`.)*

### 8. Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
Streamlit will automatically open your default browser to:
`http://localhost:8501`

---

## 📖 How to Use the Application

### 1. Adding Your Personal Documents
You have two ways to add your documents:
- **Option A (Web Upload):** In the Streamlit app, navigate to the **"📁 Document Management"** tab, upload your resume, marksheets, certificates, or project notes (`.pdf`, `.docx`, `.txt`, `.md`), and click **"⚡ Process & Index Uploaded Files"**.
- **Option B (Direct Folder):** Place your personal files directly into the `Student Knowledge Base/` folder on your machine, then click **"🔄 Re-Index 'Student Knowledge Base' Folder"** in the app.

### 2. Asking Questions
1. Switch to the **"💬 Question Answering"** tab.
2. Click any of the **Quick Sample Questions** or type your own question into the text box.
3. Click **"🔍 Ask Question"**.
4. The system will:
   - Compute normalized cosine similarity against all document chunks in FAISS.
   - Filter out any chunks with similarity below the threshold (`0.35`).
   - Prompt the local LLM with strict grounding instructions.
   - Display the generated answer, source filename badges, and an expandable list of the exact retrieved chunks with their cosine similarity scores.

### 3. Testing Non-Hallucination
Ask questions about details **not** present in your uploaded documents, for example:
- *"What is the student's exact CGPA?"* (if not listed in your resume)
- *"What is the student's favorite sports team?"*
- *"Where did the student attend kindergarten?"*

The system will strictly return:
> **"I could not find this information in the provided knowledge base."**

---

## 🔍 Demonstration Questions for Lab Viva / Evaluation

Use these questions during your lab evaluation to demonstrate the strengths of your RAG system:

| Test Case | Example Question | Expected System Behavior |
| :--- | :--- | :--- |
| **Fact Retrieval** | *"What is the student's full name?"* | Returns exact name and cites the resume filename. |
| **Technical Skills** | *"What programming languages does the student know?"* | Lists only languages explicitly mentioned in your documents. |
| **Capstone Project** | *"Which programming language was used in the final-year project?"* | Identifies project tech stack with source citation. |
| **Omission Safety** | *"What is the student's CGPA or percentage?"* (when omitted) | States information is unavailable; does not fabricate numbers. |
| **Out-of-Domain Query** | *"What is the student's favorite movie?"* | Refuses to guess; returns standard unavailable response. |
| **Offline Resilience** | Stop Ollama and ask any query. | Automatically switches to **Extractive Fallback Mode**, presenting retrieved passages without crashing. |

---

## ❓ Troubleshooting Common Problems

1. **Ollama Offline Indicator in Sidebar:**
   - Ensure the Ollama desktop app or service is running. Run `ollama serve` in a separate terminal.
   - Note: The app remains 100% functional even if Ollama is closed, switching safely to **Extractive Fallback Mode**.

2. **`UserWarning: A NumPy version ... is required`:**
   - This is a harmless warning from SciPy and does not impact embeddings or FAISS retrieval. The code automatically silences it.

3. **`Empty Index` warning when asking a question:**
   - Navigate to **Document Management** tab and click **"Process & Index Uploaded Files"** or **"Re-Index 'Student Knowledge Base' Folder"**.

4. **Port 8501 Already in Use:**
   - Run Streamlit on a custom port: `streamlit run app.py --server.port 8502`.
