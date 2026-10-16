import os
import shutil
from pathlib import Path

from dotenv import dotenv_values
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_ollama import ChatOllama

# --------------------------------------------------
# Settings
# --------------------------------------------------

# True  -> embeddings run locally, no token needed (good for testing now)
# False -> embeddings use the Hugging Face API (token required)
USE_LOCAL_EMBEDDINGS = True

# # True -> delete the old chroma_db each run so chunks are not duplicated
REBUILD_DB = True

# # Variable name used inside your .env file, e.g.  HF_TOKEN=hf_xxxxx
TOKEN_KEY = "HUGGINGFACEHUB_API_TOKEN"

MODEL_NAME = "BAAI/bge-small-en-v1.5"

PROJECT_DIR = Path(__file__).resolve().parent
ENV_FILE = PROJECT_DIR / ".env"
PDF_PATH = PROJECT_DIR / "file.pdf"
CHROMA_DIR = PROJECT_DIR / "chroma_db"


# --------------------------------------------------
# 1. Load Hugging Face token (only needed for API embeddings)
# --------------------------------------------------

config = dotenv_values(ENV_FILE)

token = (config.get("HUGGINGFACEHUB_API_TOKEN") or "").strip()

# token = (config.get(TOKEN_KEY) or os.getenv(TOKEN_KEY) or "").strip()

print("Environment file exists:", ENV_FILE.exists())
print("Token loaded:", bool(token))

if not token:
    raise ValueError(
        f"Hugging Face token missing. Add {TOKEN_KEY}=hf_xxxxx to {ENV_FILE}"
    )




# --------------------------------------------------
# 2. Check PDF path
# --------------------------------------------------

if not PDF_PATH.exists():
    raise FileNotFoundError(
        f"PDF not found: {PDF_PATH}\n"
        "Place your PDF in the project folder and name it file.pdf."
    )


# --------------------------------------------------
# 3. Load PDF
# --------------------------------------------------

print("\nLoading PDF...")

loader = PyPDFLoader(str(PDF_PATH))
documents = loader.load()

if not documents:
    raise ValueError("No pages were extracted from the PDF.")

print(f"Loaded {len(documents)} pages.")


# --------------------------------------------------
# 4. Split PDF text into chunks
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len,
)

chunks = text_splitter.split_documents(documents)

# Drop chunks that contain only whitespace
chunks = [c for c in chunks if c.page_content.strip()]

if not chunks:
    raise ValueError(
        "No text chunks were extracted. "
        "The PDF may be scanned or image-only and require OCR."
    )

print(f"Created {len(chunks)} text chunks.")


# --------------------------------------------------
# 5. Initialize embeddings
# --------------------------------------------------

print("\nInitializing embeddings...")


from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpointEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
)

# if USE_LOCAL_EMBEDDINGS:
#     # Run the embedding model locally
#     embeddings = HuggingFaceEmbeddings(
#         model_name=MODEL_NAME
#     )

# else:
#     # Use Hugging Face's hosted embedding API

MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Run embeddings locally; no Hugging Face API token needed
embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


# # Configure a separate chat model for Multi-Query Retrieval
# llm_endpoint = HuggingFaceEndpoint(
#     repo_id="HuggingFaceH4/zephyr-7b-beta",
#     task="text-generation",
#     huggingfacehub_api_token=token,
#     max_new_tokens=256,
#     do_sample=False,
# )


#used Ollama model for local embedding 
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
)

# Test the local chat model
# response = llm.invoke(
#     "Explain machine learning in two sentences."
# )

# print(response.content)

# llm = ChatHuggingFace(llm=llm_endpoint)
# print("Token length:", len(token))
# print(llm.invoke("Explain machine learning in two sentences.").content)
# --------------------------------------------------
# 6. Store embeddings and documents in ChromaDB
# --------------------------------------------------

if REBUILD_DB and CHROMA_DIR.exists():
    shutil.rmtree(CHROMA_DIR)

print("\nEmbedding chunks and storing them in ChromaDB...")


vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="pdf_knowledge",
    persist_directory=str(CHROMA_DIR),
)

print("PDF embeddings stored successfully!")


#using MMR and multiquery for better answer
mmr_retriever = vector_store.as_retriever(
    search_type="mmr",                   # <-- This enables MMR
    search_kwargs={"k": 2, "lambda_mult": 0.5}  # k = top results, lambda_mult = relevance-diversity balance
)
retriever = MultiQueryRetriever.from_llm(
    retriever=mmr_retriever,
    llm=llm,
)
query = input("Ask me a question :")
results= retriever.invoke(query)

results = results[:3]


for i, doc in enumerate(results, start=1):
    page = doc.metadata.get("page")
    page_label = page + 1 if isinstance(page, int) else "Unknown"  # pages are 0-indexed
    print(f"\n--- Result {i} ---")
    print(f"Page: {page_label}")
    print(doc.page_content[:500])

print("\nPDF ingestion and retrieval completed!")