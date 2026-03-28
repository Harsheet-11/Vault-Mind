from sentence_transformers import SentenceTransformer
from pathlib import Path
import sys
from typing import List, Dict, Optional
import numpy as np
import json

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import EMBEDDING_MODEL_NAME, EMBEDDINGS_DIR
from src.chunk_text import DocumentChunker
from src.extract_pdf import extract_text_from_pdf


class EmbeddingEngine:
    
    def __init__ (self,model_name:str = EMBEDDING_MODEL_NAME):
        self.model =SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
    def embed_chunks(self,chunks: List[dict]) -> List[dict]:
        texts = []
        for p in chunks:
            texts.append(p["text"])
            
        embeddings = self.model.encode(texts, batch_size=80)
        
        embedded_chunks = []
        for i in range(len(chunks)):
            chunk=chunks[i]
            embedding=embeddings[i]
            
            chunk["embedding"]=embedding.tolist()
            
            embedded_chunks.append(chunk)
        
        
        return embedded_chunks
    
    def save_embeddings(self, embedded_chunks: List[dict], filename: str):
        
        output_path = EMBEDDINGS_DIR / filename
        
        # Create a structured output
        output_data = {
            "model": EMBEDDING_MODEL_NAME,
            "dimension": self.dimension,
            "num_chunks": len(embedded_chunks),
            "chunks": embedded_chunks
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Saved embeddings to: {output_path}")
        print(f"   File size: {output_path.stat().st_size / 1024:.2f} KB")
        
    def load_embeddings(self, filename: str) -> List[dict]:
       
        input_path = EMBEDDINGS_DIR / filename
        
        if not input_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📂 Loaded {data['num_chunks']} chunks from {filename}")
        return data["chunks"]    
    

def pipeline_demo():
    
    print("🚀 Vault-Mind Embedding Pipeline Demo")
    print("=" * 70)
    
    # Step 1: Extract text from PDF
    print("\n📄 Step 1: Extracting text from PDF...")
    pdf_path = "data/sample.pdf"
    
    try:
        text = extract_text_from_pdf(pdf_path)
        print(f"✅ Extracted {len(text)} characters")
    except FileNotFoundError:
        print("❌ sample.pdf not found. Using demo text instead.")
        text = """
        MERGER AGREEMENT
        
        This agreement dated March 15, 2024 outlines the terms of acquisition 
        between AcquireCo and TargetCo. The purchase price is $50 million, 
        subject to working capital adjustments.
        
        FINANCIAL TERMS
        Base purchase price: $50,000,000
        Earnout potential: $10,000,000 over 3 years
        Payment: 70% cash, 30% stock
        
        EMPLOYEE RETENTION
        All employees will be retained for minimum 6 months. Key employees 
        have retention bonuses totaling $2 million.
        
        REPRESENTATIONS AND WARRANTIES
        Seller represents that all financial statements are accurate and 
        there are no undisclosed liabilities exceeding $100,000.
        """
    
    # Step 2: Chunk the text
    print("\n✂️  Step 2: Chunking text...")
    chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk_text(
        text=text,
        metadata={"source": pdf_path, "processed_date": "2024-01-15"}
    )
    print(f"✅ Created {len(chunks)} chunks")
    
    # Step 3: Generate embeddings
    print("\n🧠 Step 3: Generating embeddings...")
    embedder = EmbeddingEngine()
    embedded_chunks = embedder.embed_chunks(chunks)
    print(f"✅ Generated {len(embedded_chunks)} embeddings")
    
    # Step 4: Save embeddings
    print("\n💾 Step 4: Saving embeddings...")
    embedder.save_embeddings(embedded_chunks, "sample_embeddings.json")
    
    # Step 5: Demonstrate loading
    print("\n📂 Step 5: Testing load functionality...")
    loaded_chunks = embedder.load_embeddings("sample_embeddings.json")
    print(f"✅ Successfully loaded {len(loaded_chunks)} chunks")
    
    # Show sample
    print("\n" + "=" * 70)
    print("📊 Sample Chunk with Embedding:")
    print("=" * 70)
    sample = loaded_chunks[0]
    print(f"Text: {sample['text'][:150]}...")
    print(f"Embedding (first 10 values): {sample['embedding'][:10]}")
    print(f"Metadata: {sample['metadata']}")


if __name__ == "__main__":
    pipeline_demo()