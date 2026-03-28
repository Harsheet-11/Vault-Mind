from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

OLLAMA_BASE_URL = "http://localhost:11434"  
LLM_MODEL_NAME = "phi3:mini"  
LLM_TEMPERATURE = 0.1 
LLM_MAX_TOKENS = 500

TOP_K_RESULTS = 3
SIMILARITY_THRESHOLD = 0.2  


SYSTEM_PROMPT = """You are a legal and financial document analyst for Vault-Mind, 
a due diligence platform. Your job is to analyze documents and provide accurate, 
detailed answers based ONLY on the provided context.

Key guidelines:
- Only use information from the provided context
- If the answer isn't in the context, say "I don't have enough information"
- Cite specific details from the context
- Be precise with numbers, dates, and legal terms
- Flag any contradictions or risks you notice
"""

QUESTION_PROMPT_TEMPLATE = """Context from documents:
{context}

Question: {question}

Provide a detailed answer based on the context above. If you notice any risks, 
contradictions, or important details, highlight them.

Answer:"""