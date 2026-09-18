from google import genai
import time


def configure_gemini(api_key):
    """
    Creates and returns the Gemini API client.
    """
    return genai.Client(api_key=api_key)


def build_context(retrieved_records):
    """
    Converts retrieved records into readable context
    for Gemini.
    """

    context_parts = []
    for record in retrieved_records:
        context_parts.append(
            f"""
SOURCE {record['rank']}
Page: {record['page']}
Chunk: {record['chunk_number']}
Similarity: {record['similarity']:.4f}

Text:
{record['text']}
""")
    return "\n".join(context_parts)


def generate_answer(question, retrieved_records, gemini_client):
    """
    Generates an answer using only retrieved
    document context.
    """

    context_text = build_context(retrieved_records)

    prompt = f"""
You are a source-grounded research assistant.

Answer the question using ONLY the retrieved context
from the research paper.

Strict rules:

1. Do not use outside knowledge.
2. Do not invent technical details.
3. Do not add equations unless they appearin the retrieved context.
4. Do not claim that a method improves accuracy unless the retrieved context explicitly supports that claim.
5. Do not claim that a parameter was experimentally optimized unless the retrieved context explicitly states this. 
6. If the answer is not clearly present in the retrieved context, say:

"The retrieved sections do not provide enough
information to answer this question confidently."

7. Clearly separate direct evidence from interpretation.
8. Mention relevant page and chunk numbers.
9. Keep the answer technically accurate and concise.

User question:
{question}

Retrieved context:
{context_text}

Use this format:

## Direct answer
Give the answer using only the retrieved context.

## Evidence from the paper
Explain which statements from the retrieved sections support the answer.

## Important technical points
List only details directly supported by the context.

## Limitations
Mention information that is missing or uncertain.

## Sources
Mention page numbers and chunk numbers.
"""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )

            answer = response.text
            if answer is None:
                answer = ""
            return answer.strip()

        except Exception as error:
            error_text = str(error)

            # Retry temporary Gemini server errors
            if "503" not in error_text:
                raise
            if attempt == max_retries - 1:
                raise

            wait_time = 2 ** attempt

            print(
                f"Gemini temporarily unavailable. "
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)
    return ""