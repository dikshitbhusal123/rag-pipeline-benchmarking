import hashlib
import io
import re
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader, WikipediaLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestiondata.csv_query import answer_csv_question
from cnn_pdf import classify_pdf


# ============================================================
# 1. APP CONFIGURATION
# ============================================================

st.set_page_config(page_title="Upload-Based RAG Assistant", layout="wide")
st.title("Upload-Based RAG Assistant")
st.caption(
    "Answers are based on the file uploaded in this session. "
    "Saved ChromaDB collections are not searched."
)

PROJECT_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_models():
    embeddings_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    chat_model = ChatOllama(model="qwen2.5:3b", temperature=0)
    return embeddings_model, chat_model


embeddings, llm = load_models()


# ============================================================
# 2. FILE PROCESSING HELPERS
# ============================================================

def process_pdf(file_bytes: bytes, filename: str):
    """Extract searchable text from the uploaded PDF."""
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(file_bytes)
            temp_path = Path(temp_file.name)

        documents = PyPDFLoader(str(temp_path)).load()

        for document in documents:
            document.metadata["source"] = filename
            document.metadata["source_type"] = "Current user upload"

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )
        chunks = splitter.split_documents(documents)
        return [chunk for chunk in chunks if chunk.page_content.strip()]

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def process_csv(file_bytes: bytes, filename: str):
    """Load the uploaded CSV and make each row searchable."""
    dataframe = pd.read_csv(io.BytesIO(file_bytes))
    documents = []

    for row_index, row in dataframe.iterrows():
        row_text = "\n".join(
            f"{column}: {value}"
            for column, value in row.items()
            if pd.notna(value)
        )
        if row_text.strip():
            documents.append(
                Document(
                    page_content=row_text,
                    metadata={
                        "source": filename,
                        "source_type": "Current user upload",
                        "row_number": int(row_index) + 1,
                    },
                )
            )

    return dataframe, documents


def run_cnn_on_uploaded_pdf(file_bytes: bytes):
    """Run the CNN on the actual PDF uploaded in this session."""
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(file_bytes)
            temp_path = Path(temp_file.name)

        return classify_pdf(temp_path), None

    except Exception as error:
        return [], str(error)

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


# ============================================================
# 3. UPLOAD INTERFACE AND PER-UPLOAD PROCESSING
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a PDF or CSV file",
    type=["pdf", "csv"],
)

if uploaded_file is None:
    st.info(
        "Upload a PDF or CSV to start. The app will use that upload, "
        "not your saved ChromaDB data."
    )
    st.stop()

file_bytes = uploaded_file.getvalue()
signature = hashlib.sha256(
    uploaded_file.name.encode("utf-8") + b"\0" + file_bytes
).hexdigest()

# Process a new file once, rather than on every Streamlit rerun.
if st.session_state.get("file_signature") != signature:
    st.session_state["file_signature"] = signature
    st.session_state["file_name"] = uploaded_file.name
    st.session_state["upload_index"] = None
    st.session_state["csv_df"] = None
    st.session_state["cnn_results"] = []
    st.session_state["cnn_error"] = None
    st.session_state["processing_error"] = None
    st.session_state["chat_history"] = []

    try:
        if uploaded_file.name.lower().endswith(".pdf"):
            with st.spinner("Extracting text from the uploaded PDF..."):
                chunks = process_pdf(file_bytes, uploaded_file.name)

            # CNN runs automatically for each newly uploaded PDF.
            with st.spinner("CNN is classifying the pages of your uploaded PDF..."):
                cnn_results, cnn_error = run_cnn_on_uploaded_pdf(file_bytes)

            st.session_state["cnn_results"] = cnn_results
            st.session_state["cnn_error"] = cnn_error
            dataframe = None

        else:
            with st.spinner("Reading the uploaded CSV..."):
                dataframe, chunks = process_csv(file_bytes, uploaded_file.name)

        # Text may be absent in scanned/image-only PDFs. CNN predictions can
        # still be shown and used even if there are no searchable text chunks.
        if chunks:
            with st.spinner("Preparing search for this uploaded file..."):
                st.session_state["upload_index"] = InMemoryVectorStore.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                )

        st.session_state["csv_df"] = dataframe

    except Exception as error:
        st.session_state["processing_error"] = str(error)
        st.error(f"Could not process the uploaded file: {error}")
        st.stop()

st.success(f"Current file: {st.session_state['file_name']}")

if st.session_state.get("processing_error"):
    st.error(st.session_state["processing_error"])

# Show the CNN results for the currently uploaded PDF.
if uploaded_file.name.lower().endswith(".pdf"):
    with st.expander("CNN page predictions", expanded=True):
        cnn_error = st.session_state.get("cnn_error")
        if cnn_error:
            st.error(f"CNN analysis failed: {cnn_error}")

        cnn_results = st.session_state.get("cnn_results", [])
        if cnn_results:
            for result in cnn_results:
                st.write(
                    f"**Page {result.get('page', '?')}** — "
                    f"{result.get('category', 'Unknown')} "
                    f"({result.get('confidence', 0):.1%} confidence)"
                )
        elif not cnn_error:
            st.info("The CNN returned no predictions for this PDF.")

# Optional Wikipedia is external information, not saved ChromaDB data.
st.subheader("Optional external source")
use_wikipedia = st.checkbox(
    "Also search Wikipedia when answering questions",
    value=False,
)


# ============================================================
# 4. RETRIEVAL HELPERS
# ============================================================

def retrieve_uploaded_file(question: str):
    upload_index = st.session_state.get("upload_index")
    if upload_index is None:
        return []
    return upload_index.similarity_search(question, k=4)


def retrieve_wikipedia(question: str):
    loader = WikipediaLoader(
        query=question,
        lang="en",
        load_max_docs=2,
        doc_content_chars_max=6000,
    )
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    return splitter.split_documents(documents)


# ============================================================
# 5. ANSWER USING UPLOADED FILE + CNN + OPTIONAL WIKIPEDIA
# ============================================================

def answer_question(question: str):
    context_parts = []
    source_names = []
    uploaded_name = st.session_state.get("file_name", "Uploaded file")
    csv_dataframe = st.session_state.get("csv_df")

    # A. Exact lookups use only the CSV uploaded in the current session.
    if csv_dataframe is not None:
        csv_answer = answer_csv_question(csv_dataframe, question)
        if csv_answer is not None:
            return str(csv_answer), [f"{uploaded_name} (exact CSV lookup)"]

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
                "I couldn't find an exact matching record in the uploaded CSV. "
                "Check the ID or column name.",
                [uploaded_name],
            )

    # B. Search the uploaded file (PDF or CSV). This is OUTSIDE the CSV
    #    block above, so it runs for PDFs too.
    uploaded_docs = retrieve_uploaded_file(question)

    for document in uploaded_docs:
        page = document.metadata.get("page")
        row_number = document.metadata.get("row_number")

        if isinstance(page, int):
            location = f", page {page + 1}"
        elif isinstance(row_number, int):
            location = f", row {row_number}"
        else:
            location = ""

        context_parts.append(
            "SOURCE: CURRENT USER UPLOAD\n"
            f"File: {uploaded_name}{location}\n"
            f"Content:\n{document.page_content}"
        )

    if uploaded_docs:
        source_names.append(f"Uploaded file: {uploaded_name}")

    # C. Include CNN results generated from this same uploaded PDF.
    cnn_results = st.session_state.get("cnn_results", [])
    if cnn_results:
        cnn_text = "\n".join(
            f"Page {result.get('page', '?')}: predicted class "
            f"{result.get('category', 'Unknown')}; confidence "
            f"{result.get('confidence', 0):.1%}"
            for result in cnn_results
        )
        context_parts.append(
            "SOURCE: CNN CLASSIFICATION OF CURRENT UPLOAD\n"
            "These are model predictions, not verified facts. "
            "They classify pages only and do not explain all diagram details.\n"
            f"File: {uploaded_name}\n{cnn_text}"
        )
        source_names.append(f"CNN predictions: {uploaded_name}")

    # D. Wikipedia is only used if the user explicitly enables it.
    if use_wikipedia:
        try:
            wikipedia_docs = retrieve_wikipedia(question)
            for document in wikipedia_docs[:4]:
                context_parts.append(
                    "SOURCE: WIKIPEDIA (external source)\n"
                    f"Title: {document.metadata.get('title', 'Wikipedia')}\n"
                    f"Content:\n{document.page_content}"
                )
            if wikipedia_docs:
                source_names.append("Wikipedia (external source)")
        except Exception as error:
            st.warning(f"Wikipedia search failed: {error}")

    if not context_parts:
        return (
            "I couldn't find searchable text or CNN predictions that answer "
            "this question in the current upload. If this is a scanned PDF, "
            "OCR may be needed to extract its text.",
            [],
        )

    context = "\n\n---\n\n".join(context_parts)
    prompt = f"""
You are a careful assistant answering questions about a user's current upload.

Rules:
- Prioritize information from the current uploaded file.
- Do not use or imply access to saved ChromaDB collections or other users' files.
- Wikipedia content, if present, is external and must be labelled as such.
- Do not invent facts, values, citations, objects, defects, or diagram labels.
- CNN results are predictions, not guaranteed facts.
- The CNN classifies pages into its trained classes; its output alone does not
  explain the full meaning of a diagram.
- If the context does not answer the question, say what is missing.
- Cite the uploaded filename and page number when available.
- Use the page number stored in the chunk metadata.
- Do not invent or guess page numbers.
- Do not share any of your data to user.
- If the retrieved chunks do not establish the page,
  say that the page could not be verified.
- Distinguish between pages retrieved as relevant and
  pages confirmed to contain the requested topic.

Retrieved evidence:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)
    return response.content, list(dict.fromkeys(source_names))


# ============================================================
# 6. CHAT INTERFACE
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

for message in st.session_state["chat_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption("Sources: " + ", ".join(message["sources"]))

question = st.chat_input("Ask a question about your uploaded file...")

if question:
    st.session_state["chat_history"].append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the current upload and preparing an answer..."):
            try:
                answer, sources = answer_question(question)
            except Exception as error:
                answer = f"An error occurred while answering: {error}"
                sources = []

        st.markdown(answer)
        if sources:
            st.caption("Sources: " + ", ".join(sources))

    st.session_state["chat_history"].append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )