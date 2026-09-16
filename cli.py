import argparse
import sys
from src.config import settings
from src.rag_engine import ClinicalRAGEngine
from src.data_loader import PatientDataLoader

def print_banner():
    print("=" * 70)
    print("  PATIENT-RECORD RAG PROTOTYPE (CLINICAL EHR RETRIEVAL)")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Query synthetic patient EHR records with RAG.")
    parser.add_argument("--patient", "-p", type=str, help="Patient ID (e.g. P101, P102, P103, P104, P105)")
    parser.add_argument("--query", "-q", type=str, help="Clinical inquiry text")
    parser.add_argument("--top_k", "-k", type=int, default=5, help="Number of chunks to retrieve (default: 5)")
    parser.add_argument("--list", "-l", action="store_true", help="List all synthetic patients")
    parser.add_argument("--reindex", "-r", action="store_true", help="Force re-index vector store")

    args = parser.parse_args()
    print_banner()

    loader = PatientDataLoader(settings.DATA_PATH)
    engine = ClinicalRAGEngine()

    if args.reindex:
        print("\n[*] Re-indexing ChromaDB collection from synthetic dataset...")
        count = engine.vector_store.index_patient_records(force_reload=True)
        print(f"[OK] Successfully indexed {count} clinical documents.")
        return

    if args.list:
        summaries = loader.get_all_patients_summary()
        print(f"\nFound {len(summaries)} synthetic patient records:\n")
        for s in summaries:
            print(f" * [{s['patient_id']}] {s['full_name']} | Age: {s['age']} | Diagnoses: {', '.join(s['diagnoses'])}")
        return

    if not args.patient or not args.query:
        # Interactive mode
        summaries = loader.get_all_patients_summary()
        print("\nAvailable Patients:")
        for s in summaries:
            print(f"  [{s['patient_id']}] {s['full_name']} ({s['gender']}, {s['age']} yo)")

        patient_id = args.patient or input("\nEnter Target Patient ID (e.g., P101): ").strip()
        query = args.query or input("Enter Clinical Query: ").strip()
    else:
        patient_id = args.patient
        query = args.query

    print(f"\n[*] Querying Patient {patient_id}...")
    print(f"[*] Question: {query}\n")

    res = engine.query_patient(patient_id=patient_id, query=query, top_k=args.top_k)

    print("-" * 70)
    print(f"PATIENT: {res['patient_name']} ({res['patient_id']}) | MODEL: {res['model_used']}")
    print("-" * 70)
    print("\nCLINICAL SYNTHESIS:")
    print(res["answer"])
    
    print("\n" + "-" * 70)
    print(f"RETRIEVED CITATIONS ({len(res['citations'])} sources):")
    for c in res["citations"]:
        pct = int(c.get("similarity_score", 1.0) * 100)
        print(f" * [{c['source_id']}] Date: {c['date']} | Type: {c['record_type']} ({pct}% relevance)")
    print("=" * 70)

if __name__ == "__main__":
    main()
