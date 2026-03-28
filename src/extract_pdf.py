
from pathlib import Path
from pypdf import PdfReader  


def extract_text_from_pdf(pdf_path: str) -> str:
        
   
    pdf_file = Path(pdf_path)
    
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
   
    if not pdf_file.is_file():
        raise ValueError(f"Path is not a file: {pdf_path}")
    
    try:
       
        reader = PdfReader(pdf_file)
        
        
        num_pages = len(reader.pages)
        print(f"📄 Found {num_pages} pages in the PDF")
        
        full_text = ""
        
       
        for page_num in range(num_pages):
           
            page = reader.pages[page_num]
            
           
            page_text = page.extract_text()
            
           
            full_text += f"\n--- Page {page_num + 1} ---\n"
            full_text += page_text
            
        return full_text
    
    except Exception as e:
       
        raise Exception(f"Error reading PDF: {str(e)}")


def main():
   
    pdf_path = "1.pdf"
    
    print("🚀 Vault-Mind PDF Extractor v0.1")
    print("=" * 50)
    
    try:
       
        text = extract_text_from_pdf(pdf_path)
        
      
        print("\n📝 Extracted Text:")
        print("=" * 50)
        print(text)
        
        # Print some statistics
        print("\n" + "=" * 50)
        print(f"✅ Successfully extracted {len(text)} characters")
        print(f"✅ Word count: {len(text.split())}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\n💡 Tip: Make sure you have a 'sample.pdf' in the 'data' folder")
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()