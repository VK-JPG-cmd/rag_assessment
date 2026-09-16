"""
Synthetic Clinical EHR Dataset Generator
Generates medically coherent, longitudinal synthetic patient health records
compliant with HIPAA Safe Harbor de-identification principles.
"""

import json
import random
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

CLINICAL_ARCHETYPES = [
    {
        "specialty": "Endocrinology",
        "primary_condition": "Type 2 Diabetes Mellitus with Diabetic Nephropathy",
        "icd_codes": [
            {"code": "E11.22", "description": "Type 2 diabetes mellitus with diabetic chronic kidney disease"},
            {"code": "I10", "description": "Essential (primary) hypertension"},
            {"code": "E78.5", "description": "Hyperlipidemia, unspecified"}
        ],
        "allergies": [
            {"substance": "Sulfa Drugs", "reaction": "Urticaria and facial swelling", "severity": "Moderate"},
            {"substance": "Penicillin", "reaction": "Maculopapular rash", "severity": "Mild"}
        ],
        "medication_templates": [
            {"name": "Metformin", "dosage": "500mg twice daily", "route": "Oral", "indication": "Glycemic control"},
            {"name": "Empagliflozin (Jardiance)", "dosage": "10mg daily", "route": "Oral", "indication": "Cardiorenal protection & T2D"},
            {"name": "Lisinopril", "dosage": "20mg daily", "route": "Oral", "indication": "Hypertension & Renal Protection"},
            {"name": "Atorvastatin", "dosage": "40mg daily", "route": "Oral", "indication": "Lipid management"}
        ],
        "lab_metrics": ["Hemoglobin A1c", "Estimated GFR (eGFR)", "Serum Creatinine", "Urine Albumin/Creatinine Ratio"],
        "complaints": [
            "Routine 6-month diabetic checkup and review of blood glucose logs.",
            "Complaining of mild lower extremity edema and morning fatigue.",
            "Annual nephrology follow-up to monitor proteinuria and eGFR stability."
        ]
    },
    {
        "specialty": "Cardiology",
        "primary_condition": "Coronary Artery Disease post-PCI with Stenting & Heart Failure with preserved Ejection Fraction (HFpEF)",
        "icd_codes": [
            {"code": "I25.10", "description": "Atherosclerotic heart disease of native coronary artery"},
            {"code": "I50.32", "description": "Chronic diastolic (congestive) heart failure"},
            {"code": "I48.91", "description": "Unspecified atrial fibrillation"}
        ],
        "allergies": [
            {"substance": "Aspirin", "reaction": "Severe bronchospasm and urticaria", "severity": "Severe"}
        ],
        "medication_templates": [
            {"name": "Clopidogrel (Plavix)", "dosage": "75mg daily", "route": "Oral", "indication": "Antiplatelet post-PCI"},
            {"name": "Apixaban (Eliquis)", "dosage": "5mg twice daily", "route": "Oral", "indication": "Stroke prevention in AFib"},
            {"name": "Carvedilol", "dosage": "12.5mg twice daily", "route": "Oral", "indication": "Rate control & cardioprotection"},
            {"name": "Sacubitril/Valsartan (Entresto)", "dosage": "24/26mg twice daily", "route": "Oral", "indication": "Heart failure guideline therapy"},
            {"name": "Furosemide (Lasix)", "dosage": "20mg daily as needed", "route": "Oral", "indication": "Volume overload management"}
        ],
        "lab_metrics": ["NT-proBNP", "Troponin I", "Total Cholesterol", "LDL Cholesterol", "Serum Potassium"],
        "complaints": [
            "Exertional dyspnea when climbing stairs and mild bilateral ankle swelling.",
            "Routine cardiology post-stent surveillance and ECG rhythm check.",
            "Palpitations and lightheadedness lasting 10 minutes."
        ]
    },
    {
        "specialty": "Pulmonology",
        "primary_condition": "Moderate-to-Severe Chronic Obstructive Pulmonary Disease (COPD) & Asthma Overlap",
        "icd_codes": [
            {"code": "J44.1", "description": "Chronic obstructive pulmonary disease with acute exacerbation"},
            {"code": "J45.50", "description": "Severe persistent asthma, uncomplicated"}
        ],
        "allergies": [
            {"substance": "Codeine", "reaction": "Nausea, vomiting, severe dizziness", "severity": "Moderate"}
        ],
        "medication_templates": [
            {"name": "Fluticasone/Vilanterol (Breo Ellipta)", "dosage": "100/25 mcg once daily", "route": "Inhalation", "indication": "COPD controller"},
            {"name": "Tiotropium (Spiriva Respimat)", "dosage": "2.5 mcg 2 puffs daily", "route": "Inhalation", "indication": "LAMA bronchodilator"},
            {"name": "Albuterol/Ipratropium (Combivent)", "dosage": "2 puffs every 4-6h PRN", "route": "Inhalation", "indication": "Rescue bronchodilator"}
        ],
        "lab_metrics": ["Spirometry FEV1 (% Predicted)", "FEV1/FVC Ratio", "Absolute Eosinophil Count", "Arterial Blood Gas pO2"],
        "complaints": [
            "Increased productive cough with yellow sputum and tightness in chest.",
            "Annual spirometry evaluation and inhaler technique check.",
            "Follow-up after recovering from upper respiratory viral infection."
        ]
    },
    {
        "specialty": "Rheumatology",
        "primary_condition": "Seropositive Rheumatoid Arthritis & Osteoporosis",
        "icd_codes": [
            {"code": "M05.79", "description": "Rheumatoid arthritis with rheumatoid factor of multiple sites"},
            {"code": "M81.0", "description": "Age-related osteoporosis without current pathological fracture"}
        ],
        "allergies": [
            {"substance": "Latex", "reaction": "Contact dermatitis and localized pruritus", "severity": "Mild"}
        ],
        "medication_templates": [
            {"name": "Methotrexate", "dosage": "15mg subcutaneously once weekly", "route": "Subcutaneous", "indication": "DMARD for Rheumatoid Arthritis"},
            {"name": "Folic Acid", "dosage": "1mg daily (except MTX day)", "route": "Oral", "indication": "Methotrexate toxicity prevention"},
            {"name": "Adalimumab (Humira)", "dosage": "40mg every 2 weeks", "route": "Subcutaneous", "indication": "Anti-TNF biologic therapy"},
            {"name": "Alendronate (Fosamax)", "dosage": "70mg once weekly", "route": "Oral", "indication": "Osteoporosis prevention"}
        ],
        "lab_metrics": ["C-Reactive Protein (CRP)", "Erythrocyte Sedimentation Rate (ESR)", "Rheumatoid Factor (RF)", "Anti-CCP Antibodies", "DEXA T-Score (Lumbar Spine)"],
        "complaints": [
            "Morning stiffness in bilateral wrists and MCP joints lasting > 60 minutes.",
            "Routine 3-month biologic safety monitoring (CBC, LFTs).",
            "Joint discomfort improved; inquiring about bone mineral density scan results."
        ]
    },
    {
        "specialty": "Gastroenterology",
        "primary_condition": "Crohn's Disease (Ileocolonic) & Iron Deficiency Anemia",
        "icd_codes": [
            {"code": "K50.10", "description": "Crohn's disease of large intestine without complications"},
            {"code": "D50.9", "description": "Iron deficiency anemia, unspecified"}
        ],
        "allergies": [
            {"substance": "Ciprofloxacin", "reaction": "Achilles tendon pain and rash", "severity": "Moderate"}
        ],
        "medication_templates": [
            {"name": "Ustekinumab (Stelara)", "dosage": "90mg subcutaneous every 8 weeks", "route": "Subcutaneous", "indication": "Biologic maintenance for Crohn's"},
            {"name": "Ferric Carboxymaltose (Injectafer)", "dosage": "750mg IV infusion x 2 doses", "route": "Intravenous", "indication": "Refractory iron deficiency anemia"},
            {"name": "Mesalamine", "dosage": "1.2g twice daily", "route": "Oral", "indication": "Mucosal anti-inflammatory"}
        ],
        "lab_metrics": ["Fecal Calprotectin", "Hemoglobin", "Serum Ferritin", "C-Reactive Protein (CRP)"],
        "complaints": [
            "Intermittent lower quadrant crampy abdominal pain and 3-4 loose stools daily.",
            "Fatigue, pallor, and follow-up on post-infusion iron parameters.",
            "Surveillance colonoscopy consultation and biologic drug trough level check."
        ]
    },
    {
        "specialty": "Neurology",
        "primary_condition": "Relapsing-Remitting Multiple Sclerosis (RRMS) & Neuropathic Pain",
        "icd_codes": [
            {"code": "G35", "description": "Multiple sclerosis"},
            {"code": "G62.9", "description": "Polyneuropathy, unspecified"}
        ],
        "allergies": [],
        "medication_templates": [
            {"name": "Ocrelizumab (Ocrevus)", "dosage": "600mg IV infusion every 6 months", "route": "Intravenous", "indication": "Disease-modifying therapy for RRMS"},
            {"name": "Gabapentin", "dosage": "300mg three times daily", "route": "Oral", "indication": "Neuropathic pain & paresthesias"},
            {"name": "Baclofen", "dosage": "10mg twice daily as needed", "route": "Oral", "indication": "Lower extremity spasticity"}
        ],
        "lab_metrics": ["Brain & Spine MRI Lesion Count", "Serum Neurofilament Light Chain (sNfL)", "Total Immunoglobulin G (IgG)"],
        "complaints": [
            "Tingling paresthesias in bilateral lower extremities and mild gait unsteadiness.",
            "Routine 6-month pre-infusion laboratory and neurological disability (EDSS) check.",
            "Mild heat intolerance (Uhthoff's phenomenon) during summer weather."
        ]
    }
]

FIRST_NAMES_M = ["Liam", "Noah", "Oliver", "Arthur", "Ethan", "Alexander", "Daniel", "Mateo", "Henry", "Samuel", "Julian", "Marcus", "Gabriel", "Raymond"]
FIRST_NAMES_F = ["Emma", "Olivia", "Ava", "Sophia", "Isabella", "Charlotte", "Amelia", "Harper", "Evelyn", "Abigail", "Elena", "Clara", "Maya", "Valerie"]
LAST_NAMES = ["Patel", "Vance", "Holloway", "Lin", "Miller", "Rodriguez", "Chen", "Sterling", "Kowalski", "Gallagher", "Thorne", "Jenkins", "Mercer", "O'Connor", "Dubois", "Nakamura", "Al-Mansoor"]
PHYSICIANS = [
    "Dr. Sarah Jenkins, MD (Internal Medicine)",
    "Dr. David Sterling, MD (Primary Care)",
    "Dr. Anthony Rossi, MD (Interventional Cardiology)",
    "Dr. Marcus Reed, MD (Nephrology)",
    "Dr. Robert Gallagher, MD (Pulmonary Disease)",
    "Dr. Beatrice Thorne, MD (Rheumatology)",
    "Dr. Helen Cho, MD (Gastroenterology)",
    "Dr. Gregory Vance, MD (Orthopedic Surgery)",
    "Dr. Alistair Mercer, MD (Neurology)"
]

def generate_patient(patient_num: int, base_year: int = 2021) -> Dict[str, Any]:
    p_id = f"P{100 + patient_num}"
    is_female = random.random() > 0.5
    first_name = random.choice(FIRST_NAMES_F if is_female else FIRST_NAMES_M)
    last_name = random.choice(LAST_NAMES)
    full_name = f"{first_name} {last_name}"
    
    age = random.randint(28, 78)
    birth_year = (2024 - age)
    dob = f"{birth_year}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
    blood_type = random.choice(["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"])
    pcp = random.choice(PHYSICIANS)

    # Select archetype
    archetype = random.choice(CLINICAL_ARCHETYPES)
    
    # Active diagnoses
    active_diagnoses = []
    for d in archetype["icd_codes"]:
        onset_yr = random.randint(base_year - 3, base_year)
        active_diagnoses.append({
            "code": d["code"],
            "description": d["description"],
            "onset_date": f"{onset_yr}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
        })

    # Allergies
    allergies = []
    if random.random() < 0.65 and archetype["allergies"]:
        for a in archetype["allergies"]:
            allergies.append({
                "substance": a["substance"],
                "reaction": a["reaction"],
                "severity": a["severity"],
                "recorded_date": f"{random.randint(base_year-5, base_year-1)}-0{random.randint(1,9)}-15"
            })

    # Active medications
    active_meds = []
    for m in archetype["medication_templates"]:
        prescribed_yr = random.randint(base_year, 2023)
        active_meds.append({
            "name": m["name"],
            "dosage": m["dosage"],
            "route": m["route"],
            "indication": m["indication"],
            "prescribed_date": f"{prescribed_yr}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
        })

    # Longitudinal Encounters (3-4 encounters over years)
    encounters = []
    dates = [
        f"{base_year}-03-12",
        f"{base_year+1}-08-20",
        f"{base_year+2}-11-05",
        f"{base_year+3}-04-18"
    ]
    
    for idx, enc_date in enumerate(dates, start=1):
        enc_id = f"ENC-{p_id}-{idx:02d}"
        enc_type = "Outpatient Specialist Consultation" if idx % 2 == 1 else "Routine Chronic Disease Follow-up"
        provider = random.choice(PHYSICIANS)
        complaint = archetype["complaints"][(idx - 1) % len(archetype["complaints"])]
        
        sbp = random.randint(114, 148)
        dbp = random.randint(70, 92)
        hr = random.randint(62, 86)
        bmi = round(random.uniform(22.0, 32.5), 1)
        wt = round(random.uniform(55.0, 95.0), 1)
        
        note = (
            f"Patient {full_name} ({age}yo {('Female' if is_female else 'Male')}) presented for {enc_type.lower()}. "
            f"Chief concern: {complaint} "
            f"Medication adherence was verified and tolerability is acceptable. "
            f"Physical examination revealed regular heart rate and rhythm, clear lungs, and benign abdomen. "
            f"Plan discussed with patient: continue current medical regimen, monitor symptom diary, "
            f"and repeat surveillance laboratory testing in 6 months."
        )

        encounters.append({
            "encounter_id": enc_id,
            "encounter_date": enc_date,
            "encounter_type": enc_type,
            "provider": provider,
            "chief_complaint": complaint,
            "clinical_note": note,
            "vitals": {
                "blood_pressure": f"{sbp}/{dbp}",
                "heart_rate": hr,
                "bmi": bmi,
                "weight_kg": wt
            }
        })

    # Longitudinal Lab Panels
    lab_results = []
    for metric in archetype["lab_metrics"]:
        for d in dates[:3]:
            # Generate realistic values based on metric
            if "A1c" in metric:
                val = round(random.uniform(6.4, 8.8), 1)
                unit = "%"
                ref = "< 5.7"
                interp = "Elevated / Diabetic Range" if val > 7.0 else "Controlled"
            elif "eGFR" in metric:
                val = random.randint(48, 88)
                unit = "mL/min/1.73m2"
                ref = "> 60"
                interp = "Mild-to-Moderate Reduction" if val < 60 else "Normal"
            elif "Creatinine" in metric:
                val = round(random.uniform(0.8, 1.6), 2)
                unit = "mg/dL"
                ref = "0.6 - 1.2"
                interp = "Elevated" if val > 1.2 else "Normal"
            elif "Troponin" in metric:
                val = round(random.uniform(0.01, 12.5), 2)
                unit = "ng/mL"
                ref = "< 0.04"
                interp = "Critical Elevation" if val > 0.5 else "Within Normal Limits"
            elif "LDL" in metric:
                val = random.randint(52, 175)
                unit = "mg/dL"
                ref = "< 100"
                interp = "Optimal on Statin" if val < 70 else "Elevated"
            elif "FEV1" in metric:
                val = random.randint(62, 92)
                unit = "%"
                ref = "> 80"
                interp = "Normal" if val >= 80 else "Mild-to-Moderate Obstruction"
            elif "Uric Acid" in metric:
                val = round(random.uniform(4.5, 9.6), 1)
                unit = "mg/dL"
                ref = "< 6.0"
                interp = "Target Achieved" if val < 6.0 else "Hyperuricemia"
            elif "T-Score" in metric:
                val = round(random.uniform(-3.1, -1.2), 1)
                unit = "T-score"
                ref = "> -1.0"
                interp = "Osteoporosis" if val <= -2.5 else "Osteopenia"
            elif "CRP" in metric:
                val = round(random.uniform(1.2, 18.4), 1)
                unit = "mg/L"
                ref = "< 3.0"
                interp = "Active Systemic Inflammation" if val > 5.0 else "Normal / Controlled"
            else:
                val = round(random.uniform(10.0, 95.0), 1)
                unit = "units"
                ref = "Standard Ref"
                interp = "Clinical finding noted"

            lab_results.append({
                "test_name": metric,
                "date": d,
                "value": val,
                "unit": unit,
                "reference_range": ref,
                "interpretation": interp
            })

    return {
        "patient_id": p_id,
        "demographics": {
            "full_name": full_name,
            "gender": "Female" if is_female else "Male",
            "date_of_birth": dob,
            "age": age,
            "blood_type": blood_type,
            "primary_care_physician": pcp
        },
        "active_diagnoses": active_diagnoses,
        "allergies": allergies,
        "active_medications": active_meds,
        "encounters": encounters,
        "lab_results": lab_results
    }


def generate_full_dataset(num_patients: int = 15, output_dir: str = "data") -> Dict[str, Any]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    patients = [generate_patient(i + 1) for i in range(num_patients)]

    dataset = {
        "dataset_metadata": {
            "title": "Comprehensive Synthetic Clinical EHR Benchmark Dataset",
            "version": "2.0",
            "generated_at": datetime.now().isoformat(),
            "compliance": "HIPAA Safe Harbor De-identified Synthetic Data",
            "total_patients": len(patients)
        },
        "patients": patients
    }

    # Save master JSON dataset
    json_path = out_path / "synthetic_patients.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
    print(f"[OK] Saved JSON dataset ({len(patients)} patients) -> {json_path}")

    # Export CSVs for tabular analysis & data science workflows
    _export_csvs(patients, out_path)

    return dataset


def _export_csvs(patients: List[Dict[str, Any]], out_path: Path):
    """Exports patient records into normalized relational CSV tables."""
    # 1. Patients CSV
    with open(out_path / "patients.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "full_name", "gender", "age", "date_of_birth", "blood_type", "primary_care_physician"])
        for p in patients:
            d = p["demographics"]
            writer.writerow([p["patient_id"], d["full_name"], d["gender"], d["age"], d["date_of_birth"], d["blood_type"], d["primary_care_physician"]])

    # 2. Encounters CSV
    with open(out_path / "encounters.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["encounter_id", "patient_id", "date", "type", "provider", "chief_complaint", "blood_pressure", "heart_rate", "bmi", "weight_kg", "clinical_note"])
        for p in patients:
            for enc in p["encounters"]:
                v = enc.get("vitals", {})
                writer.writerow([
                    enc["encounter_id"], p["patient_id"], enc["encounter_date"], enc["encounter_type"],
                    enc["provider"], enc["chief_complaint"], v.get("blood_pressure"), v.get("heart_rate"),
                    v.get("bmi"), v.get("weight_kg"), enc["clinical_note"]
                ])

    # 3. Labs CSV
    with open(out_path / "labs.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "test_name", "date", "value", "unit", "reference_range", "interpretation"])
        for p in patients:
            for lab in p["lab_results"]:
                writer.writerow([p["patient_id"], lab["test_name"], lab["date"], lab["value"], lab["unit"], lab["reference_range"], lab["interpretation"]])

    # 4. Medications CSV
    with open(out_path / "medications.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "name", "dosage", "route", "indication", "prescribed_date"])
        for p in patients:
            for med in p["active_medications"]:
                writer.writerow([p["patient_id"], med["name"], med["dosage"], med["route"], med["indication"], med["prescribed_date"]])

    # 5. Allergies CSV
    with open(out_path / "allergies.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "substance", "reaction", "severity", "recorded_date"])
        for p in patients:
            for al in p["allergies"]:
                writer.writerow([p["patient_id"], al["substance"], al["reaction"], al["severity"], al["recorded_date"]])

    print(f"[OK] Exported relational CSV tables: patients.csv, encounters.csv, labs.csv, medications.csv, allergies.csv")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic patient EHR datasets.")
    parser.add_argument("--count", "-n", type=int, default=15, help="Number of synthetic patient records (default: 15)")
    parser.add_argument("--output", "-o", type=str, default="data", help="Output directory path (default: data)")
    args = parser.parse_args()

    print(f"Generating {args.count} synthetic patient EHR records...")
    generate_full_dataset(num_patients=args.count, output_dir=args.output)
