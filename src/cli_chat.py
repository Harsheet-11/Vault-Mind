import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.rag_engine import RAGEngine


class ChatInterface:
    
    def __init__(self, embeddings_file: str):
        print("🔧 Initializing Vault-Mind Chat...")
        print("=" * 60)
        
        try:
            self.rag = RAGEngine(embeddings_file)
            print("\n✅ Ready! Ask your questions (type 'quit' to exit)")
            print("=" * 60)
        except Exception as e:
            print(f"\n❌ Initialization failed: {e}")
            sys.exit(1)
    
    def run(self):
        while True:
            try:
                question = input("\n💬 You: ").strip()
                
                if not question:
                    continue
                
                # Exit commands
                if question.lower() in ["quit", "exit", "q"]:
                    print("\n👋 Goodbye!")
                    break
                
                # Help
                if question.lower() == "help":
                    print("\nType any question about your documents.")
                    print("Commands: help | stats | quit")
                    continue
                
                # Stats
                if question.lower() == "stats":
                    self.show_stats()
                    continue
                
                # Ask RAG
                print("\n🤖 Vault-Mind:", end=" ")
                
                response = self.rag.ask(question)
                print("DEBUG RESULTS:", response)
                
                print(response)
            
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("👉 Make sure Ollama is running.")
    
    def show_stats(self):
        chunks = self.rag.search_engine.chunks
        
        print("\n📊 Document Stats")
        print("-" * 40)
        print(f"Chunks: {len(chunks)}")
        print(f"Total words: {sum(len(c['text'].split()) for c in chunks)}")
        print("-" * 40)


def main():
    embeddings_file = "sample_embeddings.json"
    
    if len(sys.argv) > 1:
        embeddings_file = sys.argv[1]
    
    chat = ChatInterface(embeddings_file)
    chat.run()


if __name__ == "__main__":
    main()