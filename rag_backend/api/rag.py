import os
import uuid
from typing import List, Dict, Any
from dotenv import load_dotenv

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_groq import ChatGroq


def load_pdfs(pdf_dir: str = "api/pdf") -> List[Document]:
    loader = DirectoryLoader(
        pdf_dir,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=True
    )
    return loader.load()

pdf_docs = load_pdfs()



class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(" Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not self.model:
            raise RuntimeError("Embedding model not loaded.")
        return self.model.encode(texts, show_progress_bar=True)

embedding_manager = EmbeddingManager()



class VectorStore:
    def __init__(self, collection_name: str = "pdf_documents", persist_dir: str = "chroma_db"):
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "PDF Document Collection"}
        )
        print(" ChromaDB initialized successfully.")

    def add_documents(self, docs: List[Document], embeddings: np.ndarray):
        if len(docs) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match.")

        ids, metadatas, texts, embeddings_list = [], [], [], []

        for i, (doc, emb) in enumerate(zip(docs, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata.update({"doc_index": i, "content_length": len(doc.page_content)})
            metadatas.append(metadata)

            texts.append(doc.page_content)
            embeddings_list.append(emb.tolist())

        self.collection.add(ids=ids, embeddings=embeddings_list, metadatas=metadatas, documents=texts)
        print(f" Added {len(docs)} documents to ChromaDB.")

vectorstore = VectorStore()
texts = [doc.page_content for doc in pdf_docs]
embeddings = embedding_manager.embed_texts(texts)
vectorstore.add_documents(pdf_docs, embeddings)



class RAGRetriever:
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        query_emb = self.embedding_manager.embed_texts([query])[0]
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_emb.tolist()],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

            retrieved = []
            for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
                similarity = 1 - dist
                if similarity >= score_threshold:
                    retrieved.append({"document": doc, "metadata": meta, "similarity": similarity})
            
            print(f" Retrieved {len(retrieved)} documents for query: '{query}'")
            return retrieved
        except Exception as e:
            print(f" Retrieval error: {e}")
            return []

rag_retriever = RAGRetriever(vectorstore, embedding_manager)



load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY", "your_api_key_here")
llm = ChatGroq(api_key=groq_api_key, model="gemma2-9b-it", temperature=0)



def rag_simple(query: str, retriever: RAGRetriever, llm: ChatGroq, top_k: int = 5) -> str:
    docs = retriever.retrieve(query, top_k=top_k)
    if not docs:
        return "No relevant documents found."

    context = "\n\n".join([f"Document {i+1}:\n{d['document']}" for i, d in enumerate(docs)])
    prompt = f"""
    You are an AI assistant. Use the following context to answer the question.

    Context:
    {context}

    Question: {query}
    Answer:
    """
    response = llm.invoke(prompt)
    return response.content



if __name__ == "__main__":
    print("Checking ChromaDB documents...")
    print("Total documents:", vectorstore.collection.count())
    print(vectorstore.collection.peek())
    
    query = "fitness benefits of regular exercise"
    results = rag_retriever.retrieve(query, top_k=3)
    print("Results:", results)

    answer = rag_simple(query, rag_retriever, llm, top_k=3)
    print("\nLLM Answer:\n", answer)
