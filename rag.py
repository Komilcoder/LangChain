import json
from calendar import error
from pathlib import Path
from pydoc import text
import json
import ollama
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, ValidationError
import strip
from dotenv import load_dotenv
import os
import strip
import numpy as np
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.messages import SystemMessage

load_dotenv()



def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search(question, documents, top_k=2):
    q = embed(question)
    scored = []
    for doc in documents:
        score = cosine_similarity(q, doc['embedding'])
        scored.append((score, doc))
    scored.sort(key=lambda x:x[0],reverse=True)
    return scored[:top_k]


def answer(question, documents):
    results = search(question, documents)
    context = "\n\n".join(f"[{doc['source']}]\n{doc['text']}" for score, doc in results)
    message = [
        SystemMessage("You are a business trip assistant.\n"
            "Answer the question using only the provided context.\n"
            "If the context does not contain the answer, say: I don't know based on the documents.\n"
            "Keep all numbers and names exactly as they are in the context.\n"
            ),
        HumanMessage(f"Context:\n{context}\n\nQuestion:\n{question}"),
    ]
    llm = ChatOllama(temperature=0, model="llama3.2:3b")
    response = llm.invoke(message)
    sources = [doc['source'] for score, doc in results]
    return {'answer': response.content, "sources": sources}


def load_documents(folder=Path(__file__).parent / "docs"):
    documents = []
    for path in sorted(Path(folder).glob("*.txt")):
        documents.append({"source": path.name, "text": path.read_text()})

    return documents

def embed(text):
    """embed text"""
    response = ollama.embed(model="nomic-embed-text", input=text)
    return response["embeddings"][0]


INDEX_PATH = Path(__file__).parent / "index.json"

def build_index():
    documents = load_documents()
    for doc in documents:
        doc['embedding'] = embed(doc['text'])
    INDEX_PATH.write_text(json.dumps(documents))
    return documents


def load_index():
    if INDEX_PATH.exists():
        try:
            return json.loads(INDEX_PATH.read_text())
        except json.JSONDecodeError:
            pass
    return build_index()

