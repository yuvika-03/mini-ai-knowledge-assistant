import os
from pathlib import Path

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
GEMINI_MODEL_NAME = "gemini-3.6-flash"

TOP_K = 5
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


# =========================================================
# FIND PDF
# =========================================================

def find_pdf_file():
    data_folder = Path("data")
    pdf_files = list(data_folder.glob("*.pdf"))

    if len(pdf_files) == 0:
        raise FileNotFoundError(
            "No PDF file found inside the data folder."
        )

    return str(pdf_files[0])


# =========================================================
# ASK QUESTION
# =========================================================

def ask_question(question,embedding_model,faiss_index,chunk_records,gemini_client,top_k=TOP_K):
    """
    Retrieves relevant chunks and generates
    a source-grounded answer.

    Returns:
        answer
        retrieved_records
    """

    retrieved_records = retrieve_relevant_chunks(
        question,
        embedding_model,
        faiss_index,
        chunk_records,
        top_k
    )

    answer = generate_answer(
        question,
        retrieved_records,
        gemini_client
    )

    return answer, retrieved_records


# =========================================================
# BUILD KNOWLEDGE BASE
# =========================================================

def build_knowledge_base(pdf_path):
    """
    Processes the PDF and creates the FAISS knowledge base.

    Returns:
        embedding_model
        faiss_index
        chunk_records
    """
    page_records = extract_text_from_pdf(pdf_path)

    chunk_records = create_chunks(
        page_records,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    if len(chunk_records) == 0:
        raise ValueError("No chunks were created from the PDF.")

    embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)

    vectors, chunk_records = create_embeddings(
        chunk_records,
        embedding_model
    )

    faiss_index = create_faiss_index(vectors)

    return (
        embedding_model,
        faiss_index,
        chunk_records
    )


# =========================================================
# MAIN
# =========================================================

def main():

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key or api_key.strip() == "":
        print("ERROR: GEMINI_API_KEY is missing.")
        return

    try:
        pdf_path = find_pdf_file()
        gemini_model = configure_gemini(api_key)
        (
            embedding_model,
            faiss_index,
            chunk_records
        ) = build_knowledge_base(pdf_path)

    except Exception as error:
        print(
            f"ERROR while building knowledge base: "
            f"{type(error).__name__}: {error}"
        )
        return

    print("Knowledge base ready.")

    while True:
        question = input(
            "\nAsk a question "
            "(type 'exit' to stop): "
        ).strip()

        if question.lower() == "exit":
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            answer, retrieved_records = ask_question(
                question,
                embedding_model,
                faiss_index,
                chunk_records,
                gemini_model
            )

            print("\n" + "=" * 60)
            print("AI ANSWER")
            print("=" * 60)
            print(answer)
            print("\nSources:")

            for record in retrieved_records:
                print(
                    f"Page {record['page']}, "
                    f"Chunk {record['chunk_number']}"
                )

        except Exception as error:
            print(f"\nERROR: {type(error).__name__}: {error}")


# =========================================================
# PROGRAM ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()