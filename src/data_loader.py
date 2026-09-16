import json
from pathlib import Path
from typing import List, Dict, Any, Optional

class ClinicalDocument:
    def __init__(self, doc_id: str, text: str, metadata: Dict[str, Any]):
        self.doc_id = doc_id
        self.text = text
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "metadata": self.metadata
        }

class PatientDataLoader:
    """Loads and transforms synthetic patient records into metadata-aware clinical documents."""
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.raw_data: Dict[str, Any] = {}
        self.patients: List[Dict[str, Any]] = []

    def load_data(self) -> List[Dict[str, Any]]:
        if not self.data_path.exists():
            # If not existing, create a default empty structure
            self.raw_data = {"dataset_metadata": {"title": "Synthetic EHR"}, "patients": []}
            self.patients = []
            return self.patients
        
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)
            self.patients = self.raw_data.get("patients", [])
        return self.patients

    def save_data(self):
        """Persists patients list to JSON."""
        self.raw_data["patients"] = self.patients
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.data_path, "w", encoding="utf-8") as f:
            json.dump(self.raw_data, f, indent=2)

    def add_or_update_patients(self, new_patients: List[Dict[str, Any]]) -> List[str]:
        """Inserts or updates patient records and persists to dataset file."""
        if not self.patients:
            self.load_data()

        existing_ids = {p["patient_id"].upper(): idx for idx, p in enumerate(self.patients)}
        updated_ids = []

        for np in new_patients:
            pid = np.get("patient_id", "").strip().upper()
            if not pid:
                # Auto generate ID if missing
                pid = f"P{len(self.patients) + 101}"
                np["patient_id"] = pid
            
            # Ensure proper schema defaults
            np.setdefault("demographics", {})
            np.setdefault("active_diagnoses", [])
            np.setdefault("allergies", [])
            np.setdefault("active_medications", [])
            np.setdefault("encounters", [])
            np.setdefault("lab_results", [])

            if pid in existing_ids:
                # Update
                self.patients[existing_ids[pid]] = np
            else:
                # Add
                self.patients.append(np)
                existing_ids[pid] = len(self.patients) - 1
            
            updated_ids.append(pid)

        self.save_data()
        return updated_ids

    def get_patient_by_id(self, patient_id: str) -> Dict[str, Any]:
        if not self.patients:
            self.load_data()
        for p in self.patients:
            if p["patient_id"].upper() == patient_id.upper():
                return p
        raise ValueError(f"Patient with ID '{patient_id}' not found.")

    def get_all_patients_summary(self) -> List[Dict[str, Any]]:
        if not self.patients:
            self.load_data()
        summaries = []
        for p in self.patients:
            demos = p.get("demographics", {})
            diagnoses = [d.get("description") for d in p.get("active_diagnoses", [])]
            summaries.append({
                "patient_id": p["patient_id"],
                "full_name": demos.get("full_name", f"Patient {p['patient_id']}"),
                "age": demos.get("age", "N/A"),
                "gender": demos.get("gender", "N/A"),
                "primary_care_physician": demos.get("primary_care_physician", "N/A"),
                "diagnoses_count": len(diagnoses),
                "diagnoses": diagnoses,
                "encounters_count": len(p.get("encounters", [])),
                "medications_count": len(p.get("active_medications", [])),
                "allergies_count": len(p.get("allergies", []))
            })
        return summaries

    def chunk_single_patient(self, p: Dict[str, Any]) -> List[ClinicalDocument]:
        """Converts a single patient record into granular clinical documents."""
        documents: List[ClinicalDocument] = []
        p_id = p["patient_id"]
        demos = p.get("demographics", {})
        p_name = demos.get("full_name", f"Patient {p_id}")
        dob = demos.get("date_of_birth", "Unknown")
        age = demos.get("age", "Unknown")
        gender = demos.get("gender", "Unknown")
        blood_type = demos.get("blood_type", "Unknown")
        pcp = demos.get("primary_care_physician", "Unknown")

        # 1. Demographics & Overview Chunk
        demo_text = (
            f"PATIENT DEMOGRAPHICS & PROFILE\n"
            f"Patient ID: {p_id}\n"
            f"Full Name: {p_name}\n"
            f"Age: {age} | Gender: {gender} | DOB: {dob} | Blood Type: {blood_type}\n"
            f"Primary Care Physician: {pcp}\n"
        )
        documents.append(ClinicalDocument(
            doc_id=f"{p_id}_DEMOGRAPHICS",
            text=demo_text,
            metadata={
                "patient_id": p_id,
                "patient_name": p_name,
                "category": "Demographics",
                "record_type": "Patient Profile",
                "encounter_date": "2024-01-01",
                "source_id": f"{p_id}-DEMO"
            }
        ))

        # 2. Active Problem List / Diagnoses Chunk
        diagnoses = p.get("active_diagnoses", [])
        if diagnoses:
            diag_lines = [f"- [{d.get('code', 'ICD')}] {d.get('description', '')} (Onset: {d.get('onset_date', 'N/A')})" for d in diagnoses]
            diag_text = (
                f"ACTIVE DIAGNOSES / PROBLEM LIST\n"
                f"Patient ID: {p_id} ({p_name})\n"
                f"Summary of Diagnosed Conditions:\n" + "\n".join(diag_lines)
            )
            documents.append(ClinicalDocument(
                doc_id=f"{p_id}_DIAGNOSES",
                text=diag_text,
                metadata={
                    "patient_id": p_id,
                    "patient_name": p_name,
                    "category": "Diagnoses",
                    "record_type": "Active Problem List",
                    "encounter_date": "2024-01-01",
                    "source_id": f"{p_id}-DIAG"
                }
            ))

        # 3. Allergies Chunk
        allergies = p.get("allergies", [])
        if allergies:
            allergy_lines = [
                f"- Substance: {a.get('substance')} | Reaction: {a.get('reaction')} | Severity: {a.get('severity')} (Recorded: {a.get('recorded_date', 'N/A')})"
                for a in allergies
            ]
            allergy_text = (
                f"ALLERGIES & ADVERSE DRUG REACTIONS\n"
                f"Patient ID: {p_id} ({p_name})\n"
                f"Documented Allergies:\n" + "\n".join(allergy_lines)
            )
        else:
            allergy_text = (
                f"ALLERGIES & ADVERSE DRUG REACTIONS\n"
                f"Patient ID: {p_id} ({p_name})\n"
                f"Documented Allergies: No Known Drug Allergies (NKDA)."
            )
        documents.append(ClinicalDocument(
            doc_id=f"{p_id}_ALLERGIES",
            text=allergy_text,
            metadata={
                "patient_id": p_id,
                "patient_name": p_name,
                "category": "Allergies",
                "record_type": "Allergy Profile",
                "encounter_date": "2024-01-01",
                "source_id": f"{p_id}-ALLERGIES"
            }
        ))

        # 4. Active Medications Chunk
        medications = p.get("active_medications", [])
        if medications:
            med_lines = [
                f"- {m.get('name')} | Dosage: {m.get('dosage')} | Route: {m.get('route', 'Oral')} | Indication: {m.get('indication', 'General')} (Prescribed: {m.get('prescribed_date', 'N/A')})"
                for m in medications
            ]
            med_text = (
                f"ACTIVE MEDICATIONS & PRESCRIPTIONS\n"
                f"Patient ID: {p_id} ({p_name})\n"
                f"Current Prescriptions:\n" + "\n".join(med_lines)
            )
            documents.append(ClinicalDocument(
                doc_id=f"{p_id}_MEDICATIONS",
                text=med_text,
                metadata={
                    "patient_id": p_id,
                    "patient_name": p_name,
                    "category": "Medications",
                    "record_type": "Medication List",
                    "encounter_date": "2024-01-01",
                    "source_id": f"{p_id}-MEDS"
                }
            ))

        # 5. Encounter Notes Chunks
        for idx, enc in enumerate(p.get("encounters", []), start=1):
            enc_id = enc.get("encounter_id", f"ENC-{p_id}-{idx:02d}")
            enc_date = enc.get("encounter_date", "Undated")
            enc_type = enc.get("encounter_type", "Clinical Encounter")
            provider = enc.get("provider", "Clinical Provider")
            complaint = enc.get("chief_complaint", "Routine Evaluation")
            note = enc.get("clinical_note", "")
            vitals = enc.get("vitals", {})
            vitals_str = ", ".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in vitals.items()]) if vitals else "Not recorded"

            enc_text = (
                f"CLINICAL ENCOUNTER NOTE\n"
                f"Patient ID: {p_id} ({p_name})\n"
                f"Date: {enc_date} | Encounter ID: {enc_id}\n"
                f"Encounter Type: {enc_type}\n"
                f"Provider: {provider}\n"
                f"Chief Complaint: {complaint}\n"
                f"Vitals: {vitals_str}\n"
                f"Clinical Progress Note:\n{note}"
            )
            documents.append(ClinicalDocument(
                doc_id=f"{p_id}_{enc_id}",
                text=enc_text,
                metadata={
                    "patient_id": p_id,
                    "patient_name": p_name,
                    "category": "Encounters",
                    "record_type": enc_type,
                    "encounter_date": enc_date,
                    "provider": provider,
                    "source_id": enc_id
                }
            ))

        # 6. Lab Results Chunks
        labs = p.get("lab_results", [])
        if labs:
            tests_grouped: Dict[str, List[Dict[str, Any]]] = {}
            for lab in labs:
                t_name = lab.get("test_name", "General Lab")
                tests_grouped.setdefault(t_name, []).append(lab)

            for test_name, test_entries in tests_grouped.items():
                test_entries_sorted = sorted(test_entries, key=lambda x: str(x.get("date", "")))
                entry_lines = [
                    f"- Date: {e.get('date', 'N/A')} | Value: {e.get('value')} {e.get('unit', '')} (Ref: {e.get('reference_range', 'N/A')}) - {e.get('interpretation', 'Documented')}"
                    for e in test_entries_sorted
                ]
                latest_date = test_entries_sorted[-1].get("date", "2024-01-01") if test_entries_sorted else "2024-01-01"
                lab_text = (
                    f"LABORATORY & DIAGNOSTIC PANEL: {test_name}\n"
                    f"Patient ID: {p_id} ({p_name})\n"
                    f"Historical Measurements & Trend:\n" + "\n".join(entry_lines)
                )
                clean_test_id = "".join([c if c.isalnum() else "_" for c in test_name])
                documents.append(ClinicalDocument(
                    doc_id=f"{p_id}_LAB_{clean_test_id}",
                    text=lab_text,
                    metadata={
                        "patient_id": p_id,
                        "patient_name": p_name,
                        "category": "Labs",
                        "record_type": f"Lab: {test_name}",
                        "encounter_date": latest_date,
                        "source_id": f"{p_id}-LAB-{clean_test_id}"
                    }
                ))

        return documents

    def chunk_patient_records(self, patients_list: Optional[List[Dict[str, Any]]] = None) -> List[ClinicalDocument]:
        """Converts structured patient records into searchable clinical documents."""
        targets = patients_list if patients_list is not None else self.patients
        if not targets:
            self.load_data()
            targets = self.patients

        all_docs: List[ClinicalDocument] = []
        for p in targets:
            all_docs.extend(self.chunk_single_patient(p))
        return all_docs
