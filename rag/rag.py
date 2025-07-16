import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama

# 1. Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Load and split document into chunks
def load_document(filepath, chunk_size=100):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    # Simple chunking by sentence (or every N words)
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# 3. Create FAISS index
def create_faiss_index(docs):
    embeddings = embedder.encode(docs)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

# 4. Retrieve top-k relevant chunks
def retrieve_context(query, index, docs, k=3):
    query_vec = embedder.encode([query])
    D, I = index.search(np.array(query_vec), k)
    return [docs[i] for i in I[0]]

# 5. RAG using Ollama
def rag_generate(query, context, model='llama3'):
    prompt = f"""You are a helpful assistant.

Use the following context to answer the question:
{chr(10).join(context)}

Question: {query}
Answer:"""
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

# 6. Main function to run RAG with file input
def run_rag_with_file(file_path, query):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    documents = load_document(file_path)
    index, _ = create_faiss_index(documents)
    context = retrieve_context(query, index, documents)
    answer = rag_generate(query, context)
    return answer

# 7. Example usage
if __name__ == "__main__":
    filepath = "insurance.json"  # Replace with your .txt file path
    question = input("Enter your question: ")
    result = run_rag_with_file(filepath, question)
    print("\nAnswer:\n", result)
