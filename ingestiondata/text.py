
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


embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


def load_wikipedia(query: str):
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
    return [chunk for chunk in chunks if chunk.page_content.strip()]


if __name__ == "__main__":
    query = input("Enter a Wikipedia topic: ").strip()

    # Step 1: Create chunks first
    chunks = load_wikipedia(query)

    if chunks:
        # Step 2: Connect to the text collection
        vector_store = Chroma(
            collection_name="text_knowledge",
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )

        # Step 3: Save the chunks
        vector_store.add_documents(documents=chunks)

        print(f"Added {len(chunks)} chunks to text_knowledge.")
    else:
        print("No chunks to save.")