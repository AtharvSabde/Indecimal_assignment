import os
import pickle
import requests
import numpy as np
from typing import List, Dict

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# =============================
# ENV SETUP
# =============================
load_dotenv()

# =============================
# DOCUMENT LOADER
# =============================
class DocumentLoader:
    def load(self, path: str) -> str:
        _, ext = os.path.splitext(path)

        if ext.lower() in [".txt", ".md"]:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()

        elif ext.lower() == ".pdf":
            try:
                import PyPDF2
                text = ""
                with open(path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text += page.extract_text() or ""
                return text
            except ImportError:
                raise ImportError("Install PyPDF2 for PDF support")

        else:
            raise ValueError(f"Unsupported file format: {ext}")

# =============================
# MARKDOWN HEADER CHUNKER
# =============================
class MarkdownHeaderChunker:
    """
    Chunks markdown by headers (##), then optionally applies
    RecursiveCharacterTextSplitter for size constraints
    """

    def __init__(self, chunk_size=250, overlap=50):
        # Define headers to split on - focusing on ## (Header 2)
        self.headers_to_split_on = [
            ("##", "Header 2"),
        ]
        
        # Initialize the markdown splitter
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False  # Keep headers in content for context
        )
        
        # Initialize character splitter for size constraints
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def chunk_text(self, text: str, is_markdown: bool = True) -> List[Dict]:
        """
        Returns list of dicts with 'content' and 'metadata' keys
        """
        if is_markdown:
            # First split by markdown headers
            md_header_splits = self.markdown_splitter.split_text(text)
            
            # Then apply character-level splitting to respect chunk_size
            splits = self.text_splitter.split_documents(md_header_splits)
            
            # Convert LangChain Documents to simple dicts
            result = []
            for doc in splits:
                result.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata
                })
            return result
        else:
            # For non-markdown files, just use recursive splitter
            chunks = self.text_splitter.split_text(text)
            return [{'content': chunk, 'metadata': {}} for chunk in chunks]

# =============================
# VECTOR STORE (COSINE + CACHE)
# =============================
class VectorStore:
    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_path="embeddings.pkl"
    ):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

        # Cosine similarity
        self.index = faiss.IndexFlatIP(self.dim)

        self.cache_path = cache_path
        self.chunks = []
        self.metadata = []

    def build(self, chunks: List[str], metadata: List[Dict]):
        if os.path.exists(self.cache_path):
            print("Loading cached embeddings...")
            with open(self.cache_path, "rb") as f:
                embeddings, self.chunks, self.metadata = pickle.load(f)
        else:
            print("Generating embeddings...")
            embeddings = self.model.encode(chunks, show_progress_bar=True)
            embeddings = np.array(embeddings).astype("float32")
            faiss.normalize_L2(embeddings)

            with open(self.cache_path, "wb") as f:
                pickle.dump((embeddings, chunks, metadata), f)

            self.chunks = chunks
            self.metadata = metadata

        self.index.add(embeddings)
        print(f"FAISS index ready with {self.index.ntotal} vectors")

    def search(self, query: str, top_k=5):
        q_emb = self.model.encode([query]).astype("float32")
        faiss.normalize_L2(q_emb)

        scores, indices = self.index.search(q_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "chunk": self.chunks[idx],
                "score": float(score),
                "metadata": self.metadata[idx]
            })
        return results

# =============================
# OLLAMA LLM (GEMMA 3)
# =============================
class LLMGenerator:
    """
    Local Ollama LLM (100% offline, grounded RAG)
    """

    def __init__(self, model="gemma3:1b"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def generate(self, question: str, docs: List[Dict]) -> str:
        context = "\n\n".join([
            f"[Source: {d['metadata'].get('source', 'Unknown')} | "
            f"Section: {d['metadata'].get('Header 2', 'N/A')}]\n"
            f"{d['chunk']}"
            for d in docs
        ])

        prompt = f"""
ROLE:
You are a retrieval-augmented assistant. Your task is to answer questions
using ONLY the information explicitly present in the provided document chunks.

STRICT RULES (MUST FOLLOW):
1. Use ONLY the text in the context below.
2. Do NOT use outside knowledge or assumptions.
3. Do NOT infer missing details.
4. If the answer is not fully present, respond EXACTLY with:
   "I could not find this information in the provided documents."
5. If multiple chunks are relevant, combine them carefully without adding new facts.
6. Prefer bullet points for clarity when listing factors or steps.
7. Cite the source chunk(s) using [Source | Section] notation when possible.

CONTEXT DOCUMENTS:
{context}

QUESTION:
{question}

ANSWER:
"""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.15
                
            }
        }

        response = requests.post(self.url, json=payload)
        response.raise_for_status()

        return response.json()["response"]

# =============================
# RAG PIPELINE
# =============================
class RAGPipeline:
    def __init__(self, num_chunks=3):
        """
        Initialize RAG Pipeline
        
        Args:
            num_chunks: Number of document chunks to retrieve (default: 3)
        """
        self.loader = DocumentLoader()
        self.chunker = MarkdownHeaderChunker(chunk_size=500, overlap=50)
        self.store = VectorStore()
        self.llm = LLMGenerator(model="gemma3:1b")
        self.num_chunks = num_chunks

    def ingest(self, files: List[str]):
        all_chunks = []
        all_metadata = []

        for path in files:
            print(f"\nLoading: {path}")
            text = self.loader.load(path)
            
            # Check if it's a markdown file
            is_markdown = path.lower().endswith('.md')
            
            # Get chunks with metadata
            chunk_dicts = self.chunker.chunk_text(text, is_markdown=is_markdown)
            
            print(f"  → Generated {len(chunk_dicts)} chunks")

            for i, chunk_dict in enumerate(chunk_dicts):
                all_chunks.append(chunk_dict['content'])
                
                # Merge file metadata with chunk metadata
                combined_metadata = {
                    "source": os.path.basename(path),
                    "chunk_id": i,
                    **chunk_dict['metadata']  # Add header metadata
                }
                all_metadata.append(combined_metadata)

        self.store.build(all_chunks, all_metadata)

    def query(self, question: str):
        retrieved = self.store.search(question, top_k=self.num_chunks)
        
        # Always display retrieved context
        print("\n" + "="*80)
        print("RETRIEVED CONTEXT (Document Chunks)")
        print("="*80)
        for i, doc in enumerate(retrieved, 1):
            print(f"\n[CHUNK {i}]")
            print(f"  Source: {doc['metadata'].get('source', 'Unknown')}")
            print(f"  Section: {doc['metadata'].get('Header 2', 'N/A')}")
            print(f"  Similarity Score: {doc['score']:.4f}")
            print(f"  Content:")
            print(f"  {'-'*76}")
            # Display full chunk content with indentation
            for line in doc['chunk'].split('\n'):
                print(f"  {line}")
            print(f"  {'-'*76}")
        
        print("\n" + "="*80)
        print("GENERATING ANSWER...")
        print("="*80)
        
        answer = self.llm.generate(question, retrieved)
        
        print("\n" + "="*80)
        print("FINAL GENERATED ANSWER")
        print("="*80)
        print(answer)
        print("="*80 + "\n")
        
        return answer

# =============================
# MAIN
# =============================
def main():
    
    rag = RAGPipeline(num_chunks=3)

    docs = ["doc1.md", "doc2.md", "doc3.md"]
    docs = [d for d in docs if os.path.exists(d)]

    if not docs:
        print("No documents found.")
        return

    rag.ingest(docs)

    print("\n" + "="*60)
    print("RAG SYSTEM READY")
    print("="*60)
    print("Type your question to query (or 'exit' to quit)")
    print("="*60)

    while True:
        q = input("\n📝 Ask a question: ").strip()
        
        if q.lower() in ["exit", "quit"]:
            break
        
        rag.query(q)


if __name__ == "__main__":
    main()