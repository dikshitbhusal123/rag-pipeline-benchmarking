
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import pandas as pd
from ingestiondata.csv_query import answer_csv_question



PROJECT_DIR = Path(__file__).resolve().parent
CHROMA_DIR = PROJECT_DIR / "chroma_db"
CSV_PATH = PROJECT_DIR / "ingestiondata" / "Social_Network_Ads.csv"
csv_df = pd.read_csv(CSV_PATH)

print(f"CSV loaded: {len(csv_df)} rows")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

COLLECTION_NAMES = [
    "pdf_knowledge",
    "csv_knowledge",
    "text_knowledge",
]

vector_stores = []

for name in COLLECTION_NAMES:
    try:
        store = Chroma(
            collection_name=name,
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
        )
        vector_stores.append((name, store))
    except Exception as error:
        print(f"Could not load {name}: {error}")


# --------------------------------------------------
# 2. Local LLM
# --------------------------------------------------

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
)


# --------------------------------------------------
# 3. Automatic Wikipedia retrieval
# --------------------------------------------------

def retrieve_from_wikipedia(question: str):
    try:
        print("\nSearching Wikipedia...")

        loader = WikipediaLoader(
            query=question,
            lang="en",
            load_max_docs=2,
            doc_content_chars_max=6000,
        )

        documents = loader.load()

        if not documents:
            print("No Wikipedia pages found.")
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=150,
        )

        chunks = splitter.split_documents(documents)

        return [
            chunk for chunk in chunks
            if chunk.page_content.strip()
        ]

    except Exception as error:
        print(f"Wikipedia retrieval failed: {error}")
        return []


# --------------------------------------------------
# 4. Search local ChromaDB collections
# --------------------------------------------------

def retrieve_from_local_data(question: str):
    results = []

    for collection_name, store in vector_stores:
        try:
            documents = store.similarity_search(
                question,
                k=2,
            )

            for document in documents:
                results.append((collection_name, document))

        except Exception as error:
            print(f"Search error in {collection_name}: {error}")

    return results


# --------------------------------------------------
# 5. Generate an answer using retrieved context
# --------------------------------------------------


def answer_question(question: str):

    # 1. Exact CSV lookup first
    csv_answer = answer_csv_question(csv_df, question)

    if csv_answer is not None:
        print("\nAnswer:")
        print(csv_answer)
        return

    # Never let semantic retrieval guess a specific ID lookup
    is_record_lookup = (
        re.search(
            r"\b(id|user|employee|student|customer)\b",
            question,
            re.IGNORECASE,
        )
        and re.search(r"\d+", question)
    )

    if is_record_lookup:
        print("\nNo exact CSV record was found for that question.")
        return

    # 2. Search local PDF, CSV, and text collections in Chroma
    local_results = retrieve_from_local_data(question)

    # 3. Search Wikipedia
    try:
        wikipedia_chunks = retrieve_from_wikipedia(question)
    except Exception as error:
        print(f"\nWikipedia search failed: {error}")
        wikipedia_chunks = []

    context_parts = []

    # Local sources: PDF, CSV, and text
    for collection_name, doc in local_results:
        source = doc.metadata.get(
            "source",
            doc.metadata.get("title", collection_name),
        )

        context_parts.append(
            f"Source type: Local document\n"
            f"Collection: {collection_name}\n"
            f"Source: {source}\n"
            f"Content:\n{doc.page_content}"
        )

    # Wikipedia sources
    for doc in wikipedia_chunks:
        title = doc.metadata.get("title", "Wikipedia")
        source = doc.metadata.get("source", "Wikipedia")

        context_parts.append(
            f"Source type: Wikipedia\n"
            f"Title: {title}\n"
            f"Source: {source}\n"
            f"Content:\n{doc.page_content}"
        )

    # 4. Check whether we have any context
    if not context_parts:
        print(
            "\nNo relevant information could be retrieved "
            "from your local data or Wikipedia."
        )
        return

    # 5. Build the prompt for your existing LLM
    context = "\n\n".join(context_parts)

    prompt = f"""
You are a helpful assistant answering questions using retrieved sources.

Rules:
- Answer using the context provided below.
- Do not invent facts or numbers.
- For exact CSV record lookups, use the exact lookup system instead.
- If the context does not contain the answer, say so.
- Explain the answer clearly.

Context:
{context}

Question:
{question}

Answer:
"""

    # Keep your existing ChatOllama model
    response = llm.invoke(prompt)

    print("\nAnswer:")
    print(response.content)

    print("\nSources:")
    for collection_name, doc in local_results:
        print(
            f"- {collection_name}: "
            f"{doc.metadata.get('source', 'Local document')}"
        )

    for doc in wikipedia_chunks:
        print(
            f"- Wikipedia: "
            f"{doc.metadata.get('title', 'Wikipedia')}"
        )



# --------------------------------------------------
# 6. Run the application
# --------------------------------------------------

if __name__ == "__main__":
    print("\nUniversal RAG Assistant")
    print("Local documents + Wikipedia retrieval")
    print("Type 'exit' to quit.")

    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if question:
            answer_question(question)
