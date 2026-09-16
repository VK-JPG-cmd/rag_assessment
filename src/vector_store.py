import chromadb
from typing import List, Dict, Any, Optional
from src.config import settings
from src.data_loader import PatientDataLoader, ClinicalDocument
from src.embeddings import PatientRecordEmbeddingFunction

class PatientVectorStore:
    """Manages the ChromaDB vector database with strict patient metadata filtering."""

    COLLECTION_NAME = "patient_clinical_records"

    def __init__(self, persist_dir: str = None):
        self.persist_dir = str(persist_dir or settings.CHROMA_DB_DIR)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.embedding_fn = PatientRecordEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def index_patient_records(self, force_reload: bool = False) -> int:
        """Loads and indexes all synthetic clinical documents into ChromaDB."""
        doc_count = self.collection.count()
        if doc_count > 0 and not force_reload:
            return doc_count

        if force_reload and doc_count > 0:
            self.client.delete_collection(self.COLLECTION_NAME)
            self.collection = self.client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )

        loader = PatientDataLoader(settings.DATA_PATH)
        documents: List[ClinicalDocument] = loader.chunk_patient_records()
        return self.upsert_patient_documents(documents)

    def upsert_patient_documents(self, documents: List[ClinicalDocument]) -> int:
        """Upserts a list of clinical documents into ChromaDB collection."""
        if not documents:
            return self.collection.count()

        ids = [doc.doc_id for doc in documents]
        texts = [doc.text for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        batch_size = 25
        for i in range(0, len(documents), batch_size):
            end = i + batch_size
            self.collection.upsert(
                ids=ids[i:end],
                documents=texts[i:end],
                metadatas=metadatas[i:end]
            )

        return self.collection.count()

    def search(
        self,
        query: str,
        patient_id: str,
        n_results: int = 5,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes a similarity search with strict patient_id metadata filtering.
        Guarantees zero cross-patient data leakage.
        """
        if self.collection.count() == 0:
            self.index_patient_records()

        clean_patient_id = patient_id.strip().upper()
        if category:
            where_filter = {
                "$and": [
                    {"patient_id": clean_patient_id},
                    {"category": category}
                ]
            }
        else:
            where_filter = {"patient_id": clean_patient_id}

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        matched_items: List[Dict[str, Any]] = []
        if not results or not results["documents"] or not results["documents"][0]:
            return matched_items

        docs = results["documents"][0]
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
        ids = results["ids"][0] if results.get("ids") else [""] * len(docs)

        for doc_id, doc_text, meta, dist in zip(ids, docs, metas, distances):
            similarity = round(max(0.0, 1.0 - float(dist)), 4) if dist is not None else 1.0
            matched_items.append({
                "doc_id": doc_id,
                "text": doc_text,
                "metadata": meta,
                "distance": dist,
                "similarity_score": similarity
            })

        return matched_items
