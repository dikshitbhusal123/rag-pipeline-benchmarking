from langchain_community.document_loaders import CSVLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma

from pathlib import Path
import sys

MODEL_NAME = "BAAI/bge-small-en-v1.5"
model = HuggingFaceEmbeddings()
REBUILD=False

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from document_loader import vector_store, mmr_retriever, retriever

ENV_FILE = PROJECT_DIR / ".env"
PDF_PATH = PROJECT_DIR / "file.pdf"
CHROMA_DIR = PROJECT_DIR / "chroma_db"


loader = CSVLoader(file_path='Social_Network_Ads.csv')

docs = loader.load()
if not docs:
    print("No document were extrace")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    
)
chunks = text_splitter.split_documents(docs)
if not chunks:
    raise ValueError("no chunks were extract")

print(f"Created {len(chunks)} text chunks.")

embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)




CHROMA_DIR.mkdir(parents=True, exist_ok=True)

print("\nConnecting to CSV collection...")



vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    collection_name="csv_knowledge",
    persist_directory=str(CHROMA_DIR),
    )

print("\nLoading existing ChromaDB...")

vector_store = Chroma(
    collection_name="csv_knowledge",
    embedding_function=embedding,
    persist_directory=str(CHROMA_DIR),
)

print("ChromaDB ready!")


#using MMR and multiquery for better answer
mmr_retriever = vector_store.as_retriever(
    search_type="mmr",                   # <-- This enables MMR
    search_kwargs={"k": 2, "lambda_mult": 0.5}  # k = top results, lambda_mult = relevance-diversity balance
)
retriever = MultiQueryRetriever.from_llm(
    retriever=mmr_retriever,
    llm=llm,
)
# --------------------------------------------------
# 7. Interactive question answering
# --------------------------------------------------

if __name__ == "__main__":
    query = input("Ask me a question: ")
    results = retriever.invoke(query)[:3]

    for i, doc in enumerate(results, start=1):
        page = doc.metadata.get("page")
        page_label = page + 1 if isinstance(page, int) else "Unknown"

        print(f"\n--- Result {i} ---")
        print(f"Page: {page_label}")
        print(doc.page_content[:500])