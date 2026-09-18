# Mini AI Knowledge Assistant

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green)
![Gemini](https://img.shields.io/badge/Google-Gemini-orange)
![RAG](https://img.shields.io/badge/AI-RAG-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

A Retrieval-Augmented Generation (RAG) application that allows users to ask questions about research papers and receive source-grounded answers.

## Overview

The Mini AI Knowledge Assistant processes a PDF research paper, divides the extracted text into smaller chunks, converts the chunks into semantic embeddings, and stores them in a FAISS vector index.

When a user asks a question, the system:

1. Converts the question into an embedding.
2. Searches the FAISS index for semantically similar chunks.
3. Retrieves the top-K relevant passages.
4. Provides the retrieved passages as context to Gemini.
5. Generates an answer grounded in the retrieved document content.
6. Displays the source page and chunk information.

## Architecture

User
↓
Streamlit Interface
↓
PDF Processing
↓
Text Chunking
↓
Sentence Transformer Embeddings
↓
FAISS Vector Search
↓
Top-K Relevant Chunks
↓
Gemini
↓
Grounded Answer + Sources

## Technologies Used

- Python
- Streamlit
- Sentence Transformers
- FAISS
- Gemini API
- PyPDF
- LangChain Text Splitters
- NumPy

## RAG Pipeline

### 1. Document Processing

The uploaded PDF is processed using PyPDF. Text is extracted page by page and cleaned before further processing.

### 2. Chunking

The extracted text is divided into overlapping chunks using RecursiveCharacterTextSplitter.

Current configuration:

- Chunk size: 1200
- Chunk overlap: 200

The overlap helps preserve context between neighboring chunks.

### 3. Embeddings

Each text chunk is converted into a numerical vector using:

`all-MiniLM-L6-v2`

These vectors represent the semantic meaning of the text.

### 4. Vector Retrieval

FAISS is used to efficiently search the embedding space.

The current implementation uses normalized embeddings with inner-product similarity, which is equivalent to cosine similarity for normalized vectors.

### 5. Generation

The retrieved chunks are passed to Gemini as context.

The generation prompt instructs the model to:

- Use only retrieved document context
- Avoid unsupported claims
- Identify missing information
- Provide source page and chunk references

## Features

- PDF upload
- Automatic document processing
- Semantic search
- Top-K retrieval
- Source-grounded answers
- Page and chunk traceability
- Streamlit interface
- Gemini-based response generation
- Basic error handling and retry handling

## Project Structure

```text
mini-ai-knowledge-assistant/
│
├── app.py
├── streamlit_app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── rag/
│   ├── __init__.py
│   ├── pdf_processor.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   └── generator.py
│
└── data/