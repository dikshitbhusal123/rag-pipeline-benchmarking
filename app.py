from pathlib import Path
import hashlib
import tempfile
import re

import pandas as pd
import streamlit as st

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import (
    WikipediaLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestiondata.csv_query import answer_csv_question
from cnn_pdf import classify_pdf


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent
CHROMA_DIR = PROJECT_DIR / "chroma_db"
CSV_PATH = PROJECT_DIR / "ingestiondata" / "Social_Network_Ads.csv"

csv_df = pd.read_csv(CSV_PATH)

st.set_page_config(
    page_title="Universal RAG Assistant",
    layout="wide",
)

st.title("Universal RAG Assistant")
st.write("PDF, CNN image classification, CSV, ChromaDB, Wikipedia, and Ollama")


# ==================================================
# 2. EMBEDDINGS AND LOCAL LLM
# ==================================================

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource
def load_llm():
    return ChatOllama(
        model="qwen2.5:3b",
        temperature=0,
    )


embeddings = load_embeddings()
llm = load_llm()


# ==================================================
# 3. LOAD EXISTING CHROMA COLLECTIONS
# ==================================================

COLLECTION_NAMES = [
    "pdf_knowledge",
    "csv_knowledge",
    "text_knowledge",
]


@st.cache_resource
def load_vector_stores():
    stores = []

    for name in COLLECTION_NAMES:
        try:
            store = Chroma(
                collection_name=name,
                persist_directory=str(CHROMA_DIR),
                embedding_function=embeddings,
            )
            stores.append((name, store))
        except Exception as error:
            print(f"Could not load {name}: {error}")

    return stores


vector_stores = load_vector_stores()


# ==================================================
# 4. WIKIPEDIA RETRIEVAL
# ==================================================

def retrieve_from_wikipedia(question: str):
    try:
        loader = WikipediaLoader(
            query=question,
            lang="en",
            load_max_docs=2,
            doc_content_chars_max=6000,
        )

        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=150,
        )

        return splitter.split_documents(documents)

    except Exception as error:
        print(f"Wikipedia retrieval failed: {error}")
        return []


# ==================================================
# 5. LOCAL CHROMA RETRIEVAL (saved collections only)
# ==================================================

def retrieve_from_local_data(question: str):
    results = []

    for collection_name, store in vector_stores:
        try:
            documents = store.similarity_search(question, k=2)

            for document in documents:
                results.append((collection_name, document))

        except Exception as error:
            print(f"Search error in {collection_name}: {error}")

    return results


# ==================================================
# 5b. UPLOADED PDF RETRIEVAL (current upload only)
# ==================================================

def retrieve_uploaded_file(question: str, k: int = 4):
    """Search the vector index built from the currently uploaded PDF."""
    upload_index = st.session_state.get("upload_index")

    if upload_index is None:
        return []

    try:
        return upload_index.similarity_search(question, k=k)
    except Exception as error:
        print(f"Uploaded-file search failed: {error}")
        return []


# ==================================================
# 6. PROCESS A USER-UPLOADED PDF WITH CNN AND RAG
# ==================================================

def process_uploaded_pdf(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    file_signature = (uploaded_file.name, len(file_bytes), file_hash)

    # Avoid processing the same upload again on every Streamlit rerun.
    if st.session_state.get("file_signature") == file_signature:
        return

    st.session_state["file_signature"] = file_signature
    st.session_state["file_name"] = uploaded_file.name
    st.session_state["cnn_results"] = []
    st.session_state["cnn_error"] = None
    st.session_state["upload_index"] = None

    temp_pdf_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
        ) as temp_pdf:
            temp_pdf.write(file_bytes)
            temp_pdf_path = Path(temp_pdf.name)

        # A. Run the CNN automatically.
        with st.spinner("CNN is analyzing PDF pages..."):
            cnn_results = classify_pdf(temp_pdf_path)

        st.session_state["cnn_results"] = cnn_results

        # B. Extract PDF text for RAG.
        with st.spinner("Extracting PDF text for RAG..."):
            loader = PyPDFLoader(str(temp_pdf_path))
            documents = loader.load()

            for document in documents:
                document.metadata["source"] = uploaded_file.name

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150,
            )

            chunks = splitter.split_documents(documents)

            if chunks:
                # Unique collection name so old uploads never leak in.
                st.session_state["upload_index"] = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    collection_name=f"upload_{file_hash}",
                )

        st.session_state["messages"] = []

    except Exception as error:
        st.session_state["cnn_error"] = str(error)
        st.error(f"Could not process PDF: {error}")

    finally:
        if temp_pdf_path is not None:
            temp_pdf_path.unlink(missing_ok=True)


# ==================================================
# 7. SHOW CNN PREDICTIONS
# ==================================================

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"],
)

if uploaded_file is not None:
    process_uploaded_pdf(uploaded_file)

    with st.expander("CNN page predictions", expanded=True):
        if st.session_state.get("cnn_error"):
            st.error(st.session_state["cnn_error"])

        cnn_results = st.session_state.get("cnn_results", [])

        if cnn_results:
            for result in cnn_results:
                st.write(
                    f"**Page {result['page']}** — "
                    f"{result['category']} "
                    f"({result['confidence']:.1%} confidence)"
                )
        elif not st.session_state.get("cnn_error"):
            st.info("No CNN predictions were returned.")


# ==================================================
# 8. ANSWER QUESTIONS USING RAG + CNN
# ==================================================

def answer_question(question: str):
    context_parts = []
    source_names = []
    uploaded_name = st.session_state.get("file_name", "uploaded file")

    # A. Exact CSV lookup first.
    if csv_df is not None:
        csv_answer = answer_csv_question(csv_df, question)

        if csv_answer is not None:
            return csv_answer, ["Social_Network_Ads.csv (exact CSV lookup)"]

        # Prevent semantic search from guessing specific record values.
        is_record_lookup = (
            re.search(
                r"\b(id|user|employee|student|customer)\b",
                question,
                re.IGNORECASE,
            )
            and re.search(r"\d+", question)
        )

        if is_record_lookup:
            return (
                "I couldn't find an exact matching record in the CSV. "
                "Check the ID or column name.",
                ["Social_Network_Ads.csv"],
            )

    # B. Search the uploaded PDF.
    uploaded_docs = retrieve_uploaded_file(question)

    for document in uploaded_docs:
        page = document.metadata.get("page")
        page_text = f", page {page + 1}" if isinstance(page, int) else ""
        context_parts.append(
            "SOURCE: CURRENT USER UPLOAD\n"
            f"File: {uploaded_name}{page_text}\n"
            f"Content:\n{document.page_content}"
        )

    if uploaded_docs:
        source_names.append(f"Uploaded file: {uploaded_name}")

    # C. Retrieve relevant local documents.
    local_results = retrieve_from_local_data(question)

    for collection_name, document in local_results:
        source = document.metadata.get(
            "source",
            document.metadata.get("title", collection_name),
        )

        context_parts.append(
            f"Source type: Local document\n"
            f"Collection: {collection_name}\n"
            f"Source: {source}\n"
            f"Content:\n{document.page_content}"
        )

        source_names.append(f"{collection_name}: {source}")

    # D. Retrieve Wikipedia information.
    wikipedia_chunks = retrieve_from_wikipedia(question)

    for document in wikipedia_chunks:
        title = document.metadata.get("title", "Wikipedia")
        source = document.metadata.get("source", "Wikipedia")

        context_parts.append(
            f"Source type: Wikipedia\n"
            f"Title: {title}\n"
            f"Source: {source}\n"
            f"Content:\n{document.page_content}"
        )

        source_names.append(f"Wikipedia: {title}")

    # E. Add CNN predictions to the context.
    cnn_results = st.session_state.get("cnn_results", [])

    if cnn_results:
        image_analysis = "\n".join(
            f"Page {result.get('page', '?')}: "
            f"Predicted class = {result.get('category', 'Unknown')}, "
            f"confidence = {result.get('confidence', 0):.1%}"
            for result in cnn_results
        )

        context_parts.append(
            "Source type: CNN page classification\n"
            "These are model predictions, not verified facts.\n"
            f"File: {uploaded_name}\n"
            f"{image_analysis}"
        )

        source_names.append("CNN page classification")

    if not context_parts:
        return (
            "I couldn't retrieve relevant information from your "
            "uploaded PDF, local collections, or Wikipedia.",
            [],
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
You are a helpful RAG assistant.

Rules:
- Answer using the retrieved context below.
- Do not give your own answer give which is based on the given contex.
- Do not invent facts, numbers, or document content.
- If the context does not answer the question, say so.
- CNN outputs are predictions, not verified facts.
- CNN page classification alone cannot explain a diagram's meaning.
- Clearly distinguish retrieved facts from model predictions.
- Explain the answer in simple, clear language.

Retrieved context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content, list(dict.fromkeys(source_names))


# ==================================================
# 9. CHAT INTERFACE
# ==================================================

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.write(f"- {source}")


question = st.chat_input("Ask a question about your data or PDF")

if question:
    st.session_state["messages"].append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching sources and generating answer..."):
            try:
                answer, sources = answer_question(question)
                st.markdown(answer)

                if sources:
                    with st.expander("Sources"):
                        for source in sources:
                            st.write(f"- {source}")

                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except Exception as error:
                st.error(f"Could not generate answer: {error}")