import numpy as np
from sentence_transformers import SentenceTransformer


def load_embedding_model(model_name="all-MiniLM-L6-v2"):
    """
    Loads the SentenceTransformer embedding model.
    """
    return SentenceTransformer(model_name)


def create_embeddings(chunk_records, embedding_model):
    """
    Creates normalized embeddings for all valid chunks.

    Returns:
        vectors
        successful_records
    """

    all_vectors = []
    successful_records = []

    for index, record in enumerate(chunk_records):
        text = record.get("text", "")

        if not isinstance(text, str):
            text = str(text)

        text = text.replace("\x00", "")
        text = " ".join(text.split())

        if text.strip() == "":
            continue

        try:
            vector = embedding_model.encode(
                [text],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            all_vectors.append(vector[0])
            record["text"] = text
            successful_records.append(record)

        except Exception as error:
            print(
                f"Warning: skipped chunk {index} during embedding: "
                f"{type(error).__name__}: {error}"
            )

    if len(all_vectors) == 0:
        raise ValueError("No embeddings were created.")

    vectors = np.array(all_vectors).astype("float32")
    return vectors, successful_records