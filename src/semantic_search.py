import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import EMBEDDING_MODEL_NAME, TOP_K_RESULTS, SIMILARITY_THRESHOLD


class SemanticSearchEngine:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        print(f"🔍 Loading search model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.chunks: List[Dict] = []

    def load_embeddings(self, embedded_chunks: List[Dict]):
        if not isinstance(embedded_chunks, list):
            raise TypeError(
                f"Expected list of dicts, got {type(embedded_chunks).__name__}"
            )
        
        self.chunks = embedded_chunks
        print(f"📚 Loaded {len(self.chunks)} chunks for search")

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def search(
        self, query: str, top_k: int = TOP_K_RESULTS, threshold: float = SIMILARITY_THRESHOLD
    ) -> List[Dict]:
        if not self.chunks:
            raise ValueError("No embeddings loaded. Call load_embeddings() first.")

        query_embedding = self.model.encode([query])[0]
        results = []

        for chunk in self.chunks:
            sim = self.cosine_similarity(query_embedding, np.array(chunk["embedding"]))
            if sim >= threshold:
                results.append({
                    "text": chunk["text"],
                    "metadata": chunk.get("metadata", {}),
                    "similarity": sim
                })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]


def demo_search():
    import json

    print("🚀 Semantic Search Demo")
    print("=" * 80)
    
    from config.settings import EMBEDDINGS_DIR
    
    embeddings_path = EMBEDDINGS_DIR / "sample_embeddings.json"

    try:
        with open(embeddings_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if isinstance(data, dict) and "chunks" in data:
            chunks = data["chunks"]
            print(f"✅ Model: {data.get('model', 'unknown')}")
            print(f"✅ Embedding dimension: {data.get('dimension', 'unknown')}")
            print(f"✅ Number of chunks: {data.get('num_chunks', len(chunks))}")
            
        elif isinstance(data, list):
            chunks = data
        else:
            print(f"❌ Unexpected JSON structure. Type: {type(data)}")
            return
            
    except FileNotFoundError:
        print("❌ 'sample_embeddings.json' not found. Generate embeddings first!")
        return
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        return

    search_engine = SemanticSearchEngine()
    search_engine.load_embeddings(chunks)

    queries = [
        "What is the salary?",
        "termination conditions",
        "non-compete clause",
        "how much is the bonus?"
    ]

    for query in queries:
        results = search_engine.search(query, top_k=2)
        print(f"\n🔍 Query: {query}")
        print("-" * 80)
        if not results:
            print("No results found above similarity threshold.")
        for i, r in enumerate(results, 1):
            print(f"{i}. Similarity: {r['similarity']:.3f}")
            print(f"   Text: {r['text'][:300]}...")
            if r['metadata']:
                print(f"   Metadata: {r['metadata']}")
        print("=" * 80)


if __name__ == "__main__":
    demo_search()