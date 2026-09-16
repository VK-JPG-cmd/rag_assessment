# 🩺 Patient-Record RAG Prototype

A production-grade **Clinical Retrieval-Augmented Generation (RAG)** prototype designed to retrieve historical medical information from synthetic Electronic Health Records (EHR) and generate temporally accurate, grounded clinical answers with explicit source citations.

---

## 🌟 Key Architecture & Capabilities

1. **Synthetic Clinical EHR Dataset (`data/synthetic_patients.json`)**:
   - 5 diverse synthetic patient profiles with longitudinal encounters, active problem lists, allergies, prescriptions, and historical lab panels (HbA1c, eGFR, Troponin, Lipid panels, Spirometry, DEXA scores).
2. **Metadata-Aware Isolation & Smart Chunking (`src/data_loader.py`)**:
   - Chunks retain crucial clinical headers: `patient_id`, `encounter_date`, `record_type`, `category`, and `source_id`.
   - Guaranteed **Zero Cross-Patient Data Leakage** via strict metadata filtering.
3. **Vector Database (`src/vector_store.py`)**:
   - Powered by ChromaDB with cosine similarity search.
   - Embeddings support **Google Gemini (`models/gemini-embedding-001`)** with deterministic local fallbacks for 100% offline reliability.
4. **Clinical RAG Orchestrator (`src/rag_engine.py`)**:
   - Specialized clinical system prompts enforcing zero-hallucination guardrails, chronological timeline sorting, and exact source citation tags (`[Source: ENC-P101-01]`).
5. **Interactive Clinical Web Dashboard (`src/api.py`, `static/`)**:
   - Modern glassmorphism UI with real-time patient switcher, allergy warnings, diagnosis tags, quick clinical prompt templates, and interactive vector evidence inspector.
6. **CLI & Automated Test Suite (`cli.py`, `tests/test_rag.py`)**:
   - Instant terminal interface and test suite verifying patient isolation and retrieval fidelity.

---

## 🚀 Quick Start

### 1. Installation
Clone or navigate to the project directory and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file (or use the existing template):
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3-flash-preview
EMBEDDING_MODEL=models/gemini-embedding-001
PORT=8000
HOST=127.0.0.1
```

### 3. Run the Web Application
Start the FastAPI server:
```bash
python -m src.api
```
Open your browser at **`http://127.0.0.1:8000`**.

---

## 💻 Command-Line Interface (CLI)

You can query patient records directly from your terminal:

* **List available synthetic patients:**
  ```bash
  python cli.py --list
  ```

* **Query a specific patient:**
  ```bash
  python cli.py --patient P101 --query "What is the patient's HbA1c history and how has Metformin been adjusted?"
  ```

* **Force Re-index the Vector Store:**
  ```bash
  python cli.py --reindex
  ```

---

## 🧪 Automated Testing

Run the test suite to verify data loading, patient isolation guardrails, and retrieval accuracy:
```bash
python -m unittest discover tests
```

---

## 📂 Project Structure

```
├── .env                       # Environment & API Key configuration
├── .env.example               # Configuration template
├── cli.py                     # Command-line query tool
├── data/
│   └── synthetic_patients.json # 5 detailed synthetic EHR patient records
├── requirements.txt           # Project dependencies
├── src/
│   ├── api.py                 # FastAPI server & endpoints
│   ├── config.py              # Configuration & path management
│   ├── data_loader.py         # Clinical chunking & metadata parser
│   ├── embeddings.py          # Google GenAI & local fallback embeddings
│   ├── rag_engine.py          # RAG pipeline, clinical prompts & citations
│   └── vector_store.py        # ChromaDB vector store & filtering
├── static/
│   ├── app.js                 # Dashboard frontend logic & API calls
│   ├── index.html             # Responsive clinical web UI
│   └── style.css              # Custom clinical design styling
└── tests/
    └── test_rag.py            # Automated test suite
```
