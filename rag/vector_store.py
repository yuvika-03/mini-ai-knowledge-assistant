import faiss


def create_faiss_index(vectors):
    """
    Creates a FAISS inner-product similarity index.

    Since the embeddings are normalized,
    inner product is equivalent to cosine similarity.
    """

    if vectors is None:
        raise ValueError("Vectors cannot be None.")

    if len(vectors) == 0:
        raise ValueError("Vectors list is empty.")

    embedding_dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(embedding_dimension)
    index.add(vectors)

    return index