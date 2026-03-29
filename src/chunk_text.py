from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
import sys
from pathlib import Path
import tiktoken

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


class DocumentChunker:

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoder = tiktoken.get_encoding("cl100k_base")

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lambda text:len(self.encoder.encode(text)),
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk_text(self, text: str, metadata: Optional[dict] = None) -> List[dict]:
        chunks = self.splitter.split_text(text)

        chunk_objects = []
        for i, chunk_text in enumerate(chunks):
            chunk_obj = {
                "chunk_id": i,
                "text": chunk_text,
                "char_count": len(chunk_text),
                "word_count": len(chunk_text.split()),
                "metadata": metadata or {}
            }

            chunk_objects.append(chunk_obj)

        return chunk_objects

    def chunk_documents(self, documents: List[dict]) -> List[dict]:
        all_chunks = []

        for p in documents:
            text = p.get("text", "")
            metadata = p.get("metadata", {})

            chunks = self.chunk_text(text, metadata)
            all_chunks.extend(chunks)

        return all_chunks
    
