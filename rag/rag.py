import os
import fitz  # PyMuPDF
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ollama

# Load and read supported file types
def read_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.txt':
        encodings = ['utf-8', 'latin1', 'cp1252']
        for enc in encodings:
            try:
                with open(filepath, 'r', encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError("Could not decode the file with known encodings.")
    
    elif ext == '.docx':
        doc = Document(filepath)
        return '\n'.join([para.text for para in doc.paragraphs])
    
    elif ext == '.pdf':
        text = ""
        with fitz.open(filepath) as doc:
            for page in doc:
                text += page.get_text()
        return text

    else:
        raise ValueError("Unsupported file type. Use .txt, .docx, or .pdf")

# Split text into chunks
def split_text(text, chunk_size=300):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Retrieve top chunks using semantic search
def semantic_search(chunks, query, top_k=3):
    vectorizer = TfidfVectorizer().fit(chunks + [query])
    chunk_vecs = vectorizer.transform(chunks)
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, chunk_vecs).flatten()
    top_indices = similarities.argsort()[::-1][:top_k]
    return [chunks[i] for i in top_indices]

# Query the llama3 model with retrieved context
def query_llm(context, question):
    prompt = f"""You are an insurance domain assistant. Based on the following policy records, answer the question in a professional tone.

Context:
{context}

Question:
{question}

Answer:"""

    response = ollama.chat(model='llama3', messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

# MAIN PROGRAM
if __name__ == '__main__':
    file_path = input("Enter file path (.txt, .pdf, .docx): ").strip()
    question = input("Ask your insurance-related question: ").strip()

    print("\nReading and processing file...")
    full_text = read_file(file_path)
    chunks = split_text(full_text, chunk_size=500)
    top_chunks = semantic_search(chunks, question)
    context = "\n\n".join(top_chunks)

    print("\nGenerating answer using Llama3...")
    answer = query_llm(context, question)
    print("\n🧠 Answer:\n", answer)
