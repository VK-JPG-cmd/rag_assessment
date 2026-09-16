import re
from typing import Dict, Any, List, Optional
from src.config import settings
from src.data_loader import PatientDataLoader
from src.vector_store import PatientVectorStore

CLINICAL_SYSTEM_PROMPT = """You are a highly precise, board-certified Clinical AI Assistant specializing in Electronic Health Record (EHR) retrieval and historical analysis.

YOUR CORE OBJECTIVES:
1. Answer the clinician's query accurately, objectively, and completely based SOLELY on the retrieved patient records below.
2. Maintain strict patient data fidelity. Never invent, extrapolate, or assume diagnoses, lab values, or medications not explicitly documented.
3. Be Temporally Explicit: Highlight exact encounter dates, historical trends (e.g., how a lab value or vital sign changed from past encounters to the most recent one), and active vs resolved statuses.
4. Cite Your Sources: Use precise citation tags corresponding to the source note (e.g., `[Source: ENC-P101-01 (2021-01-15)]` or `[Source: Lab: Hemoglobin A1c (2024-05-20)]`).
5. Absence of Data: If the patient's record does not contain the requested information, explicitly state: "This information is not documented in the provided medical records for this patient."

PATIENT RECORD CONTEXT:
Patient ID: {patient_id}
Patient Name: {patient_name}

=== RETRIEVED HISTORICAL CLINICAL RECORDS ===
{context}
============================================

CLINICIAN QUERY:
{query}

CLINICAL RESPONSE (Organized with clear clinical findings, temporal timeline/trends if applicable, and source citations):"""

class ClinicalRAGEngine:
    """End-to-end RAG orchestrator for querying patient medical records."""

    def __init__(self, vector_store: Optional[PatientVectorStore] = None):
        self.vector_store = vector_store or PatientVectorStore()
        self.data_loader = PatientDataLoader(settings.DATA_PATH)
        self.client = None

        if settings.GEMINI_API_KEY:
            try:
                from google import genai
                self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
            except Exception as e:
                print(f"[RAG Engine] Warning: Could not initialize Gemini client: {e}")

    def query_patient(
        self,
        patient_id: str,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes an end-to-end clinical RAG query:
        1. Validates patient existence.
        2. Retrieves relevant chunks strictly isolated to patient_id.
        3. Formats temporal context.
        4. Synthesizes answer using Gemini or local clinical summarizer.
        """
        clean_patient_id = patient_id.strip().upper()
        
        # Verify patient exists
        try:
            patient_info = self.data_loader.get_patient_by_id(clean_patient_id)
            patient_name = patient_info.get("demographics", {}).get("full_name", f"Patient {clean_patient_id}")
        except Exception:
            return {
                "patient_id": clean_patient_id,
                "patient_name": "Unknown Patient",
                "query": query,
                "answer": f"Patient with ID '{clean_patient_id}' was not found in the synthetic health record database.",
                "citations": [],
                "retrieved_evidence": [],
                "model_used": "System Error Handler"
            }

        # Retrieve relevant clinical documents
        evidence_chunks = self.vector_store.search(
            query=query,
            patient_id=clean_patient_id,
            n_results=top_k,
            category=category
        )

        if not evidence_chunks:
            return {
                "patient_id": clean_patient_id,
                "patient_name": patient_name,
                "query": query,
                "answer": f"No relevant medical records found for Patient {clean_patient_id} regarding this query.",
                "citations": [],
                "retrieved_evidence": [],
                "model_used": "Empty Context Handler"
            }

        # Build structured temporal context
        context_blocks = []
        citations_list = []

        for idx, chunk in enumerate(evidence_chunks, start=1):
            meta = chunk.get("metadata", {})
            src_id = meta.get("source_id", f"DOC-{idx}")
            rec_type = meta.get("record_type", "Clinical Document")
            enc_date = meta.get("encounter_date", "Undated")
            category_val = meta.get("category", "General")
            
            header = f"--- [Record #{idx} | Source: {src_id} | Type: {rec_type} | Date: {enc_date} | Category: {category_val}] ---"
            context_blocks.append(f"{header}\n{chunk['text']}")

            citations_list.append({
                "source_id": src_id,
                "record_type": rec_type,
                "date": enc_date,
                "category": category_val,
                "similarity_score": chunk.get("similarity_score", 1.0)
            })

        full_context = "\n\n".join(context_blocks)

        # Generate response
        prompt = CLINICAL_SYSTEM_PROMPT.format(
            patient_id=clean_patient_id,
            patient_name=patient_name,
            context=full_context,
            query=query
        )

        model_name = settings.GEMINI_MODEL
        answer_text = ""

        if self.client:
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                answer_text = response.text if hasattr(response, "text") and response.text else str(response)
            except Exception as e:
                print(f"[RAG Engine] Gemini generation encountered error: {e}. Utilizing local summarizer.")
                answer_text = self._generate_local_fallback(patient_name, query, evidence_chunks)
                model_name = "Local Clinical Summarizer (Fallback)"
        else:
            answer_text = self._generate_local_fallback(patient_name, query, evidence_chunks)
            model_name = "Local Clinical Summarizer (No API Key)"

        return {
            "patient_id": clean_patient_id,
            "patient_name": patient_name,
            "query": query,
            "answer": answer_text,
            "citations": citations_list,
            "retrieved_evidence": evidence_chunks,
            "model_used": model_name
        }

    def _generate_local_fallback(
        self,
        patient_name: str,
        query: str,
        evidence_chunks: List[Dict[str, Any]]
    ) -> str:
        """Rule-based local fallback summarizing retrieved evidence when API is unavailable."""
        lines = [
            f"**Clinical Summary for {patient_name}** *(Generated via Local Offline Retrieval)*\n",
            f"**Query**: *{query}*\n",
            "**Relevant Clinical Findings Retrieved from Records:**"
        ]
        
        for chunk in evidence_chunks:
            meta = chunk.get("metadata", {})
            src = meta.get("source_id", "Record")
            date = meta.get("encounter_date", "Date N/A")
            rec_type = meta.get("record_type", "Note")
            
            # Extract first few lines of text
            snippet = "\n".join(chunk["text"].split("\n")[:4])
            lines.append(f"\n- **[{rec_type} | {date} | Source: {src}]**:\n  {snippet}")

        lines.append("\n\n*Note: To enable dynamic generative reasoning, verify the Gemini API key in configuration.*")
        return "\n".join(lines)
