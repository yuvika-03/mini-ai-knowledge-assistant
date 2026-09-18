def retrieve_relevant_chunks(question, embedding_model, faiss_index, chunk_records, top_k=5):
    """
    Retrieves the most relevant chunks for a question
    using cosine similarity.
    """

    question_vector = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    question_vector = question_vector.astype("float32")
    actual_k = min(
        top_k,
        len(chunk_records)
    )

    similarities, indices = faiss_index.search(
        question_vector,
        actual_k
    )

    retrieved_records = []
    for rank, index in enumerate(indices[0]):
        if index < 0:
            continue
        if index >= len(chunk_records):
            continue

        record = chunk_records[index]
        retrieved_record = record.copy()
        retrieved_record["rank"] = rank + 1
        retrieved_record["similarity"] = float(
            similarities[0][rank]
        )

        retrieved_records.append(
            retrieved_record
        )
    return retrieved_records