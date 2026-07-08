import os
import json
import numpy as np
import faiss

from openai import AzureOpenAI
from docx import Document
from pypdf import PdfReader
import streamlit as st

# ---------------- Azure Config ----------------

AZURE_ENDPOINT = st.secrets["AZURE_ENDPOINT"]
DEPLOYMENT_NAME = st.secrets["DEPLOYMENT_NAME"]
AZURE_API_KEY  = st.secrets["AZURE_API_KEY"]
AZURE_API_VERSION = st.secrets["AZURE_API_VERSION"]             # your deployment name

# IMPORTANT: These must be your deployment names in Azure
EMBED_MODEL = "my-embedding"   # embedding deployment name
CHAT_MODEL = "gpt-4.1"                  # chat deployment name


client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_API_VERSION
)

# ---------------- Document Readers ----------------

def read_pdf(file_path):

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text


def read_docx(file_path):

    doc = Document(file_path)

    return "\n".join(
        [p.text for p in doc.paragraphs]
    )


def read_txt(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        return f.read()


def load_document(file_path):

    if file_path.endswith(".pdf"):
        return read_pdf(file_path)

    if file_path.endswith(".docx"):
        return read_docx(file_path)

    if file_path.endswith(".txt"):
        return read_txt(file_path)

    raise ValueError("Unsupported file")


# ---------------- Chunking ----------------

def chunk_text(text, chunk_size=300):

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):

        chunks.append(
            " ".join(
                words[i:i + chunk_size]
            )
        )

    return chunks


# ---------------- Embedding ----------------

def get_embedding(text):

    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )

    return np.array(
        response.data[0].embedding,
        dtype=np.float32
    )


# ---------------- Build Vector Store ----------------

def build_faiss_index(chunks):

    embeddings = []

    for chunk in chunks:

        embeddings.append(
            get_embedding(chunk)
        )

    vectors = np.array(embeddings)

    dimension = vectors.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(vectors)

    return index


# ---------------- Retrieve ----------------

def retrieve_relevant_chunks(
        query,
        chunks,
        index,
        top_k=3):

    qvec = get_embedding(query).reshape(1, -1)

    D, I = index.search(qvec, top_k)

    return "\n\n".join(
        chunks[idx] for idx in I[0]
    )