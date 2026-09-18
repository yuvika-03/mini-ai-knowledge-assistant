from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_chunks(page_records, chunk_size=1200, chunk_overlap=200):
    """
    Splits page text into smaller chunks.
    Each chunk stores:
    - chunk_id
    - page
    - chunk_number
    - text
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunk_records = []

    for page_record in page_records:
        page_number = page_record["page"]
        page_text = page_record["text"]
        page_chunks = text_splitter.split_text(page_text)

        for chunk_number, chunk in enumerate(page_chunks, start=1):
            if not isinstance(chunk, str):
                chunk = str(chunk)

            chunk = chunk.replace("\x00", "")
            chunk = " ".join(chunk.split())

            if chunk.strip() == "":
                continue

            chunk_records.append({
                "chunk_id": len(chunk_records),
                "page": page_number,
                "chunk_number": chunk_number,
                "text": chunk
            })

    return chunk_records