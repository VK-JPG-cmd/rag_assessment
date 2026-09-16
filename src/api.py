import json
import csv
import io
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.config import settings
from src.data_loader import PatientDataLoader
from src.vector_store import PatientVectorStore
from src.rag_engine import ClinicalRAGEngine

app = FastAPI(
    title="Patient-Record RAG Prototype API",
    description="A clinical Retrieval-Augmented Generation system for querying synthetic patient health records.",
    version=settings.VERSION
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize data and RAG components
data_loader = PatientDataLoader(settings.DATA_PATH)
vector_store = PatientVectorStore()
rag_engine = ClinicalRAGEngine(vector_store=vector_store)

@app.on_event("startup")
def startup_event():
    try:
        count = vector_store.index_patient_records(force_reload=False)
        print(f"[Startup] Vector Store indexed with {count} clinical documents.")
    except Exception as e:
        print(f"[Startup] Indexing error: {e}")

class QueryRequest(BaseModel):
    patient_id: str = Field(..., description="Target patient identifier (e.g. P101)")
    query: str = Field(..., description="Clinical inquiry or question regarding patient history")
    top_k: int = Field(default=5, ge=1, le=15, description="Number of context records to retrieve")
    category: Optional[str] = Field(default=None, description="Optional category filter (Encounters, Labs, Medications, Allergies)")

class QueryResponse(BaseModel):
    patient_id: str
    patient_name: str
    query: str
    answer: str
    citations: List[Dict[str, Any]]
    retrieved_evidence: List[Dict[str, Any]]
    model_used: str

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "chroma_docs_count": vector_store.collection.count(),
        "gemini_api_configured": bool(settings.GEMINI_API_KEY)
    }

@app.get("/api/patients")
def get_all_patients():
    """Returns summarized profiles of all available synthetic patients."""
    try:
        summaries = data_loader.get_all_patients_summary()
        return {"total": len(summaries), "patients": summaries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/patient/{patient_id}")
def get_patient_details(patient_id: str):
    """Returns the full clinical file for a specific patient."""
    try:
        patient = data_loader.get_patient_by_id(patient_id)
        return patient
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query", response_model=QueryResponse)
def query_patient_records(request: QueryRequest):
    """Executes clinical RAG retrieval and generation for a patient."""
    try:
        result = rag_engine.query_patient(
            patient_id=request.patient_id,
            query=request.query,
            top_k=request.top_k,
            category=request.category
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG processing failed: {str(e)}")

@app.post("/api/upload-dataset")
async def upload_patient_dataset(file: UploadFile = File(...)):
    """
    Accepts JSON or CSV dataset file, updates patient records,
    and indexes new clinical documents into the ChromaDB vector database.
    """
    try:
        content_bytes = await file.read()
        filename = file.filename.lower()
        new_patients = []

        if filename.endswith(".json"):
            text_str = content_bytes.decode("utf-8")
            parsed = json.loads(text_str)
            if isinstance(parsed, dict) and "patients" in parsed:
                new_patients = parsed["patients"]
            elif isinstance(parsed, list):
                new_patients = parsed
            elif isinstance(parsed, dict) and "patient_id" in parsed:
                new_patients = [parsed]
            else:
                raise ValueError("JSON file must contain a 'patients' array or a single patient object with 'patient_id'.")

        elif filename.endswith(".csv"):
            # Simple CSV ingestion (Patients table)
            text_str = content_bytes.decode("utf-8")
            reader = csv.DictReader(io.StringIO(text_str))
            for row in reader:
                pid = row.get("patient_id") or f"P{len(data_loader.patients) + 101}"
                new_patients.append({
                    "patient_id": pid,
                    "demographics": {
                        "full_name": row.get("full_name") or row.get("name") or f"Patient {pid}",
                        "gender": row.get("gender", "Unknown"),
                        "age": int(row.get("age", 45)) if str(row.get("age", "")).isdigit() else 45,
                        "date_of_birth": row.get("date_of_birth", "1980-01-01"),
                        "blood_type": row.get("blood_type", "O+"),
                        "primary_care_physician": row.get("primary_care_physician", "Attending Physician")
                    },
                    "active_diagnoses": [{"code": "Z00.00", "description": row.get("diagnosis", "Routine clinical check"), "onset_date": "2023-01-01"}] if row.get("diagnosis") else [],
                    "allergies": [{"substance": row.get("allergy"), "reaction": "Documented Reaction", "severity": "Moderate", "recorded_date": "2023-01-01"}] if row.get("allergy") else [],
                    "active_medications": [{"name": row.get("medication"), "dosage": "As prescribed", "route": "Oral", "indication": "General", "prescribed_date": "2023-01-01"}] if row.get("medication") else [],
                    "encounters": [
                        {
                            "encounter_id": f"ENC-{pid}-01",
                            "encounter_date": "2023-06-15",
                            "encounter_type": "Initial Ingested Visit",
                            "provider": row.get("primary_care_physician", "Primary Care"),
                            "chief_complaint": row.get("chief_complaint", "Health assessment & baseline evaluation"),
                            "clinical_note": row.get("clinical_note", "Patient record ingested from external dataset."),
                            "vitals": {"blood_pressure": "120/80", "heart_rate": 72, "bmi": 24.5, "weight_kg": 70.0}
                        }
                    ],
                    "lab_results": []
                })
        else:
            raise ValueError("Unsupported file format. Please upload a .json or .csv file.")

        if not new_patients:
            raise ValueError("No valid patient records were extracted from the uploaded file.")

        # Add or update in data loader & persist
        updated_ids = data_loader.add_or_update_patients(new_patients)

        # Chunk and upsert to vector store
        new_docs = []
        for p in new_patients:
            new_docs.extend(data_loader.chunk_single_patient(p))

        total_docs = vector_store.upsert_patient_documents(new_docs)

        return {
            "status": "success",
            "message": f"Successfully ingested {len(new_patients)} patient record(s).",
            "updated_patient_ids": updated_ids,
            "total_documents_indexed": total_docs,
            "total_patients": len(data_loader.patients)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Dataset upload failed: {str(e)}")

@app.get("/api/sample-template")
def get_sample_template():
    """Provides a sample JSON schema template for uploading custom patients."""
    sample = {
        "patients": [
            {
                "patient_id": "P999",
                "demographics": {
                    "full_name": "Sample Patient",
                    "gender": "Female",
                    "date_of_birth": "1985-05-20",
                    "age": 39,
                    "blood_type": "A+",
                    "primary_care_physician": "Dr. Sample Physician, MD"
                },
                "active_diagnoses": [
                    {"code": "I10", "description": "Essential (primary) hypertension", "onset_date": "2022-01-10"}
                ],
                "allergies": [
                    {"substance": "Penicillin", "reaction": "Hives and pruritus", "severity": "Moderate", "recorded_date": "2020-04-15"}
                ],
                "active_medications": [
                    {"name": "Lisinopril", "dosage": "10 mg daily", "route": "Oral", "indication": "Hypertension", "prescribed_date": "2022-01-10"}
                ],
                "encounters": [
                    {
                        "encounter_id": "ENC-P999-01",
                        "encounter_date": "2023-04-12",
                        "encounter_type": "Outpatient Consultation",
                        "provider": "Dr. Sample Physician, MD",
                        "chief_complaint": "Routine blood pressure follow-up.",
                        "clinical_note": "Blood pressure well controlled at 122/78 mmHg on Lisinopril 10mg. Patient tolerates therapy well without dry cough or angioedema.",
                        "vitals": {"blood_pressure": "122/78", "heart_rate": 70, "bmi": 24.2, "weight_kg": 65.0}
                    }
                ],
                "lab_results": [
                    {"test_name": "Serum Creatinine", "date": "2023-04-12", "value": 0.85, "unit": "mg/dL", "reference_range": "0.6 - 1.1", "interpretation": "Normal"}
                ]
            }
        ]
    }
    return sample

@app.post("/api/reindex")
def reindex_database():
    """Forces a clean re-indexing of all patient documents into ChromaDB."""
    try:
        count = vector_store.index_patient_records(force_reload=True)
        return {"status": "success", "indexed_documents": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static frontend files
if settings.STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")

@app.get("/")
def serve_index():
    index_file = settings.STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": f"{settings.PROJECT_NAME} API running. Access /docs for interactive documentation."}

if __name__ == "__main__":
    uvicorn.run("src.api:app", host=settings.HOST, port=settings.PORT, reload=True)
