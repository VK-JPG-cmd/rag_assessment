import unittest
from pathlib import Path
from src.config import settings
from src.data_loader import PatientDataLoader
from src.vector_store import PatientVectorStore
from src.rag_engine import ClinicalRAGEngine

class TestPatientRecordRAG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_loader = PatientDataLoader(settings.DATA_PATH)
        cls.vector_store = PatientVectorStore()
        cls.vector_store.index_patient_records(force_reload=False)
        cls.engine = ClinicalRAGEngine(vector_store=cls.vector_store)

    def test_data_loader_structure(self):
        """Verify synthetic patient dataset loads correctly."""
        patients = self.data_loader.load_data()
        self.assertGreaterEqual(len(patients), 5)
        
        # Test patient P101 structure
        p101 = self.data_loader.get_patient_by_id("P101")
        self.assertIn("demographics", p101)
        self.assertTrue(len(p101["demographics"]["full_name"]) > 0)
        self.assertIn("allergies", p101)
        self.assertIn("active_medications", p101)
        self.assertIn("encounters", p101)
        self.assertIn("lab_results", p101)

    def test_patient_isolation_guardrail(self):
        """Verify absolute patient data isolation (zero cross-patient leakage)."""
        # Querying specifically for P101 should ONLY return P101 chunks
        results = self.vector_store.search(
            query="clinical diagnosis and medication history",
            patient_id="P101",
            n_results=5
        )
        for r in results:
            self.assertEqual(
                r["metadata"]["patient_id"], "P101",
                f"Data leakage detected! Found {r['metadata']['patient_id']} in P101 query results."
            )

    def test_end_to_end_rag_query(self):
        """Verify end-to-end clinical query execution returns valid answer and citations."""
        response = self.engine.query_patient(
            patient_id="P101",
            query="What are the documented active conditions and prescription medications?",
            top_k=3
        )
        self.assertEqual(response["patient_id"], "P101")
        self.assertIsNotNone(response["answer"])
        self.assertGreater(len(response["answer"]), 20)
        self.assertGreater(len(response["citations"]), 0)

if __name__ == "__main__":
    unittest.main()
