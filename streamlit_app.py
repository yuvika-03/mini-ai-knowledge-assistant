import os
import tempfile
import hashlib

import streamlit as st
from dotenv import load_dotenv

from rag.pdf_processor import extract_text_from_pdf
from rag.chunker import create_chunks
from rag.embeddings import load_embedding_model, create_embeddings
from rag.vector_store import create_faiss_index
from rag.retriever import retrieve_relevant_chunks
from rag.generator import configure_gemini, generate_answer


# =========================================================
# CONFIGURATION
# =========================================================
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Mini AI Knowledge Assistant",
    page_icon="📚",
    layout="wide"
)


# =========================================================
# LOAD ENVIRONMENT
# =========================================================
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = None


# =========================================================
# CHECK API KEY
# =========================================================
if not api_key:
    st.error(
        "GEMINI_API_KEY is missing. "
    )
    st.stop()


# =========================================================
# TITLE
# =========================================================
st.title("📚 Mini AI Knowledge Assistant")
st.write(
    "Ask questions about your research paper "
    "using a source-grounded AI assistant."
)


# =========================================================
# PDF UPLOAD
# =========================================================
st.subheader("📄 Upload a research paper")
uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)


# =========================================================
# WAIT FOR PDF
# =========================================================
if uploaded_file is None:
    st.info(
        "Upload a PDF to start asking questions."
    )
    st.stop()
pdf_bytes = uploaded_file.getvalue()


# =========================================================
# DOCUMENT CHANGE DETECTION
# =========================================================
document_id = hashlib.md5(pdf_bytes).hexdigest()

if (
    "document_id" not in st.session_state
    or st.session_state.document_id != document_id
):

    st.session_state.document_id = document_id
    # Clear conversation when a new PDF is uploaded
    st.session_state.messages = []


# =========================================================
# CHAT HISTORY INITIALIZATION
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# BUILD KNOWLEDGE BASE
# =========================================================
@st.cache_resource
def build_knowledge_base(pdf_bytes):

    # -----------------------------------------------------
    # CREATE TEMPORARY PDF
    # -----------------------------------------------------
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:
        temp_file.write(pdf_bytes)
        pdf_path = temp_file.name


    try:
        # -------------------------------------------------
        # PDF TEXT EXTRACTION
        # -------------------------------------------------
        page_records = extract_text_from_pdf(
            pdf_path
        )


        # -------------------------------------------------
        # CHUNKING
        # -------------------------------------------------
        chunk_records = create_chunks(
            page_records,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )


        if len(chunk_records) == 0:
            raise ValueError(
                "No chunks were created from the PDF."
            )


        # -------------------------------------------------
        # LOAD EMBEDDING MODEL
        # -------------------------------------------------
        embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)
        vectors, chunk_records = create_embeddings(
            chunk_records,
            embedding_model
        )


        # -------------------------------------------------
        # CREATE FAISS INDEX
        # -------------------------------------------------
        faiss_index = create_faiss_index(
            vectors
        )


        # -------------------------------------------------
        # CREATE GEMINI CLIENT
        # -------------------------------------------------
        gemini_client = configure_gemini(api_key)
        return (
            embedding_model,
            faiss_index,
            chunk_records,
            gemini_client
        )
    finally:
        # -------------------------------------------------
        # DELETE TEMPORARY PDF
        # -------------------------------------------------
        if os.path.exists(pdf_path):
            os.remove(pdf_path)


# =========================================================
# PROCESS PDF
# =========================================================
with st.spinner(
    "Processing research paper..."
):
    try:
        (
            embedding_model,
            faiss_index,
            chunk_records,
            gemini_client
        ) = build_knowledge_base(pdf_bytes)

    except Exception as error:
        st.error(
            f"Error while processing PDF: "
            f"{type(error).__name__}: {error}"
        )
        st.stop()

st.success(f"📄 Active document: {uploaded_file.name}")


# =========================================================
# KNOWLEDGE BASE INFORMATION
# =========================================================
col1, col2 = st.columns(2)
with col1:
    st.metric(
        "Chunks",
        len(chunk_records)
    )

with col2:
    st.metric(
        "Top-K Retrieval",
        TOP_K
    )


# =========================================================
# CHAT SECTION
# =========================================================
st.divider()
st.subheader("💬 Ask about the paper")


# =========================================================
# DISPLAY PREVIOUS CHAT HISTORY
# =========================================================
for message in st.session_state.messages:
    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )


# =========================================================
# CHAT INPUT
# =========================================================
question = st.chat_input(
    "Ask something about the paper..."
)


# =========================================================
# PROCESS NEW QUESTION
# =========================================================
if question:
    with st.chat_message("user"):
        st.markdown(question)


    # =====================================================
    # BUILD RETRIEVAL QUERY
    # =====================================================
    retrieval_query = question
    previous_user_questions = [
        message["content"]
        for message in st.session_state.messages
        if message["role"] == "user"
    ]


    # -----------------------------------------------------
    # INCLUDE RECENT QUESTIONS FOR CONTEXT
    # -----------------------------------------------------
    if previous_user_questions:
        recent_questions = (
            previous_user_questions[-3:]
        )
        retrieval_query = (
            "Previous questions:\n"
            + "\n".join(recent_questions)
            + "\n\nCurrent question:\n"
            + question
        )


    # =====================================================
    # RETRIEVE RELEVANT CHUNKS
    # =====================================================
    try:
        retrieved_records = (
            retrieve_relevant_chunks(
                retrieval_query,
                embedding_model,
                faiss_index,
                chunk_records,
                TOP_K
            )
        )

    except Exception as error:
        with st.chat_message("assistant"):
            st.error(
                f"Retrieval error: "
                f"{type(error).__name__}: {error}"
            )
        st.stop()


    # =====================================================
    # RETRIEVAL DEBUG VIEW
    # =====================================================
    with st.expander(
        "🔎 Retrieved Context",
        expanded=False
    ):
        st.markdown(
            "**Retrieval query:**"
        )
        st.code(
            retrieval_query
        )
        st.markdown(
            f"**Retrieved {len(retrieved_records)} chunks:**"
        )

        for record in retrieved_records:
            st.markdown(
                f"### Rank {record['rank']} — "
                f"Page {record['page']} — "
                f"Chunk {record['chunk_number']}"
            )
            st.write(
                record["text"]
            )
            st.caption(
                f"Similarity: "
                f"{record['similarity']:.4f}"
            )
            st.divider()


    # =====================================================
    # GENERATE ANSWER
    # =====================================================
    with st.chat_message("assistant"):
        try:
            with st.spinner(
                "Generating answer..."
            ):
                answer = generate_answer(
                    question,
                    retrieved_records,
                    gemini_client
                )
            st.markdown(
                answer
            )


            # =================================================
            # DISPLAY SOURCES
            # =================================================
            st.markdown(
                "### 📖 Sources"
            )

            for record in retrieved_records:
                with st.expander(
                    f"Page {record['page']} • "
                    f"Chunk {record['chunk_number']}"
                ):
                    st.write(
                        record["text"]
                    )
                    st.caption(
                        f"Similarity: "
                        f"{record['similarity']:.4f}"
                    )


            # =================================================
            # SAVE SUCCESSFUL CONVERSATION
            # =================================================
            st.session_state.messages.append({
                "role": "user",
                "content": question
            })
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })


        # =====================================================
        # ERROR HANDLING
        # =====================================================
        except Exception as error:
            error_text = str(error)


            if "429" in error_text:
                st.warning(
                    "Gemini API quota has been reached. "
                    "The retrieved context is shown above. "
                    "Please try again after the quota resets."
                )


            # -------------------------------------------------
            # 503 — TEMPORARY SERVER ERROR
            # -------------------------------------------------
            elif "503" in error_text:
                st.warning(
                    "Gemini is temporarily experiencing "
                    "high demand. Please try again later."
                )


            # -------------------------------------------------
            # OTHER ERRORS
            # -------------------------------------------------
            else:
                st.error(
                    f"Error while generating answer: "
                    f"{type(error).__name__}: {error}"
                ) 