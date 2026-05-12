import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import ollama

load_dotenv()

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_store

def search_chunks(question, k=3):
    vector_store = load_vector_store()
    results = vector_store.similarity_search(question, k=k)
    return results

def answer_question(question):
    chunks = search_chunks(question)

    context = "\n\n".join([doc.page_content for doc in chunks])

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I could not find that in the document."

Context:
{context}

Question: {question}

Answer:"""

    response = ollama.chat(
        model="llama3.2:1b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

if __name__ == "__main__":
    question = "What is name of person which is mention in document?"
    print(f"\nQuestion: {question}\n")
    answer = answer_question(question)
    print(f"Answer: {answer}")