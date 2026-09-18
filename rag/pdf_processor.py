from pypdf import PdfReader

def clean_text(text):
    """
    Cleans extracted PDF text.
    """

    if text is None: return ""
    if not isinstance(text, str):
        text = str(text)

    text = text.replace("\x00", "")
    text = " ".join(text.split())

    return text.strip()


def extract_text_from_pdf(pdf_path):
    """
    Extracts text from every page of the PDF.

    Returns:
        A list of dictionaries containing
        page number and text.
    """

    reader = PdfReader(pdf_path)
    page_records = []

    for page_number, page in enumerate(reader.pages, start=1):
        extracted_text = page.extract_text()
        extracted_text = clean_text(extracted_text)

        if extracted_text == "": continue

        page_records.append({
            "page": page_number,
            "text": extracted_text
        })

    if len(page_records) == 0:
        raise ValueError("No readable text was found in the PDF.")

    return page_records