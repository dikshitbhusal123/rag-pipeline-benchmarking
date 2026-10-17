
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpointEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
)
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma

from pathlib import Path
import sys


REBUILD=False

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from document_loader import vector_store, mmr_retriever, retriever

ENV_FILE = PROJECT_DIR / ".env"
PDF_PATH = PROJECT_DIR / "file.pdf"
CHROMA_DIR = PROJECT_DIR / "chroma_db"

MODEL_NAME = "BAAI/bge-small-en-v1.5"


# load_dotenv()

model = HuggingFaceEmbeddings()





def load_wikipedia(query):
    loader = WikipediaLoader(
        query=query,
        lang="en",
        load_max_docs=3,
        doc_content_chars_max=10000,
    )

    documents = loader.load()

    if not documents:
        print("No topics found.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(documents)
    chunks = [c for c in chunks if c.page_content.strip()]

    return chunks


embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
    )




llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
)

# vector_store = Chroma.from_documents(
#     documents=chunks,
#     embedding=embeddings,
#     collection_name="pdf_knowledge",
#     persist_directory=str(CHROMA_DIR),
# )

mmr_retriever = vector_store.as_retriever(
    search_type="mmr",                   # <-- This enables MMR
    search_kwargs={"k": 2, "lambda_mult": 0.5}  # k = top results, lambda_mult = relevance-diversity balance
)
retriever = MultiQueryRetriever.from_llm(
    retriever=mmr_retriever,
    llm=llm,
)


if __name__ == "__main__":
    query = input("Enter a Wikipedia topic: ").strip()

    if query:
        chunks = load_wikipedia(query)

        if chunks:
            vector_store.add_documents(documents=chunks)
            print(f"Added {len(chunks)} chunks to ChromaDB.")

    # Ask questions using the stored PDF and Wikipedia chunks
    while True:
        question = input(
            "\nAsk a question (or type 'exit'): "
        ).strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        # Retrieve relevant chunks
        docs = vector_store.similarity_search(question, k=4)

        if not docs:
            print("No relevant documents found.")
            continue

        context = "\n\n".join(
            doc.page_content for doc in docs
        )

        # Generate an answer using your local LLM
        prompt = f"""
Answer the question using the context below.
If the context does not contain enough information,
say so instead of inventing facts.

Context:
{context}

Question: {question}

Answer:
"""

        response = llm.invoke(prompt)

        print("\nAnswer:")
        print(response.content)

        print("\nSources:")
        for doc in docs:
            print(
                "-",
                doc.metadata.get("title")
                or doc.metadata.get("source", "Unknown")
            )
