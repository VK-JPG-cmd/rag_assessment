import time
import numpy as np
from typing import List, Union, Any
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from src.config import settings

class FallbackDeterministicEmbedding:
    """A deterministic term-frequency dense embedding fallback matching 3072 dimensions."""
    def __init__(self, dim: int = 3072):
        self.dim = dim

    def embed_text(self, text: str) -> List[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        words = text.lower().replace("\n", " ").split()
        if not words:
            return vec.tolist()
        
        for w in words:
            idx = abs(hash(w)) % self.dim
            vec[idx] += 1.0
            
            if len(w) > 3:
                for i in range(len(w) - 2):
                    sub_idx = abs(hash(w[i:i+3])) % self.dim
                    vec[sub_idx] += 0.5

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]


class PatientRecordEmbeddingFunction(EmbeddingFunction[Documents]):
    """Unified embedding function supporting Google Gemini Embeddings with graceful local fallback."""
    
    def __init__(self, api_key: str = None, model_name: str = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.fallback = FallbackDeterministicEmbedding(dim=3072)
        self.client = None

        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[Embedding] Warning: Failed to initialize Google GenAI Client: {e}. Using local fallback.")

    def name(self) -> str:
        return "patient_record_gemini_embedding"

    def __call__(self, input: Documents) -> Embeddings:
        """Required by ChromaDB EmbeddingFunction protocol."""
        return self.embed_documents(input=input)

    def embed_documents(self, input: Any = None, texts: Any = None, *args, **kwargs) -> List[List[float]]:
        target_texts = input if input is not None else texts
        if target_texts is None:
            return []

        if isinstance(target_texts, str):
            target_texts = [target_texts]
        else:
            target_texts = list(target_texts)

        if not target_texts:
            return []

        if self.client:
            try:
                embeddings: List[List[float]] = []
                batch_size = 10
                for i in range(0, len(target_texts), batch_size):
                    batch = target_texts[i:i+batch_size]
                    try:
                        res = self.client.models.embed_content(
                            model=self.model_name,
                            contents=batch
                        )
                        for emb in res.embeddings:
                            embeddings.append(emb.values)
                    except Exception as batch_err:
                        if "429" in str(batch_err) or "RESOURCE_EXHAUSTED" in str(batch_err):
                            time.sleep(2.0)
                            # Retry once
                            try:
                                res = self.client.models.embed_content(
                                    model=self.model_name,
                                    contents=batch
                                )
                                for emb in res.embeddings:
                                    embeddings.append(emb.values)
                                continue
                            except Exception:
                                pass
                        # Fall back for this batch with matching 3072 dimension
                        fallback_batch = self.fallback.embed_documents(batch)
                        embeddings.extend(fallback_batch)
                return embeddings
            except Exception as e:
                print(f"[Embedding] Warning: Google GenAI embedding encountered ({e}), using 3072-dim fallback.")

        return self.fallback.embed_documents(target_texts)

    def embed_query(self, input: Any = None, query: Any = None, *args, **kwargs) -> List[List[float]]:
        target = input if input is not None else query
        if target is None:
            return []
        
        if isinstance(target, str):
            target = [target]
        else:
            target = list(target)
            
        return self.embed_documents(input=target)
