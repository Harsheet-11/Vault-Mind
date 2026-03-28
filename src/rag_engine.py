import requests
import numpy as np
from typing import List, Dict
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config.settings import ( OLLAMA_BASE_URL,LLM_MODEL_NAME,LLM_TEMPERATURE,EMBEDDING_MODEL_NAME,TOP_K_RESULTS, SYSTEM_PROMPT,LLM_MAX_TOKENS,QUESTION_PROMPT_TEMPLATE)

from src.semantic_search import SemanticSearchEngine
from src.embed_chunks import EmbeddingEngine


class RAGEngine:
    def __init__(self, embeddings_file: str, ollama_url: str = OLLAMA_BASE_URL, model_name: str = LLM_MODEL_NAME):
        self.ollama_url = ollama_url
        self.model_name = model_name
        
        # Load embeddings
        chunks = EmbeddingEngine().load_embeddings(embeddings_file)
        self.search_engine = SemanticSearchEngine()
        self.search_engine.load_embeddings(chunks)
    
    def generate_answer(self, question: str, context: str) -> str:
        prompt = QUESTION_PROMPT_TEMPLATE.format(context=context, question=question)
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
            "options": {"temperature": LLM_TEMPERATURE, "num_predict": LLM_MAX_TOKENS}
        }
        try:
            res = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=120)
            res.raise_for_status()
            return res.json()["response"]
        except Exception as e:
            return f"❌ LLM call failed: {e}"
    
    def ask(self, question: str):
        # Step 1: Retrieve top chunks
        results = self.search_engine.search(question)
        if not results:
            return "No relevant info found."
        
        # Step 2: Combine context
        context = "\n\n".join([r["text"] for r in results])
        
        # Step 3: Generate answer
        return self.generate_answer(question, context)

# Demo
if __name__ == "__main__":
    rag = RAGEngine("sample_embeddings.json")
    question = "What is the purchase price in the agreement?"
    answer = rag.ask(question)
    print("💬 Question:", question)
    print("📝 Answer:", answer)
