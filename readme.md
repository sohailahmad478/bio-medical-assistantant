Biomedical Paper Gap Assistant — RAG

A Streamlit application that lets you upload a biomedical signal-processing research paper and use RAG (Retrieval-Augmented Generation) with FAISS, Sentence Transformers, PyMuPDF, and Groq to identify possible future research gaps.

Files

Only two application files are required:

app.py
requirements.txt

How it works

Research Paper PDF
        ↓
PyMuPDF text extraction
        ↓
Small text chunks
        ↓
Sentence Transformers embeddings
        ↓
FAISS vector index
        ↓
Retrieve relevant chunks
        ↓
Groq GPT-OSS-20B
        ↓
Possible research gaps
        ↓
MS research topics + roadmap

The important improvement is that the complete paper is NOT sent to Groq. Only the most relevant retrieved chunks are sent for each question. This greatly reduces token usage and helps avoid 413 TPM errors.

Run locally

Open a terminal in the project folder:

pip install -r requirements.txt
streamlit run app.py

Streamlit Cloud deployment

Create a GitHub repository.

Upload:

app.py

requirements.txt

Create a Streamlit Community Cloud app from the repository.

In the Streamlit app settings, open Secrets.

Add:

GROQ_API_KEY = "YOUR_GROQ_API_KEY"

Do NOT put your real API key inside app.py or commit it to GitHub.

What the app can produce

Paper evidence

Author-stated limitations/future work

AI-inferred possible research gaps

MS-level research opportunities

Recommended research topic

Cautious novelty estimate

General dataset directions

Step-by-step research roadmap

Downloadable analysis

Important

This is a research assistant, not a medical diagnostic system.

Scanned/image-only PDFs may not work because the basic version uses PDF text extraction rather than OCR.
