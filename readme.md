Biomedical Research Gap Assistant

Upload a biomedical signal-processing research paper and use Groq to identify possible future research gaps.

Features

PDF upload directly in Streamlit

Extracts text with PyMuPDF

Identifies what the paper actually did

Extracts author-stated limitations

Extracts author-stated future work

Generates additional possible research gaps

Ranks opportunities for an MS project

Suggests one research title

Suggests datasets and evaluation metrics

Creates an 8-12 step research roadmap

Provides a Streamlit deployment direction

Files

app.py
requirements.txt
readme.md

Run locally

pip install -r requirements.txt
streamlit run app.py

Groq API key

For local use, you can enter the key in the sidebar.

For Streamlit Community Cloud, use App Settings -> Secrets:

GROQ_API_KEY = "your_groq_api_key"

Do NOT upload your API key to GitHub.

Streamlit recommends storing secrets outside the repository and supports app secrets for deployed applications.

Deploy on Streamlit Community Cloud

Create a GitHub repository.

Upload:

app.py

requirements.txt

readme.md

Open Streamlit Community Cloud.

Select the GitHub repository.

Select app.py.

Add the Groq key under Secrets.

Deploy.

Research workflow

Research Paper PDF
       ↓
PyMuPDF text extraction
       ↓
Paper understanding
       ↓
Author limitations
       ↓
Author future work
       ↓
AI-inferred possible gaps
       ↓
Gap ranking
       ↓
MS research topic
       ↓
Dataset
       ↓
Baseline
       ↓
Proposed method
       ↓
Robustness / explainability
       ↓
Evaluation
       ↓
Streamlit demo
       ↓
Paper + thesis

Important distinction

The application separates:

Author-stated future work

Something explicitly mentioned by the paper's authors.

AI-inferred possible gap

A research possibility inferred from limitations, methodology, evaluation, or missing experiments.

An AI-inferred gap is NOT automatically a novel contribution.

Before using a gap in an MS proposal, perform a second literature review using recent papers.

Example MS direction

For an ECG paper, possible directions may include:

robust ECG classification under noise

cross-subject generalization

explainable deep learning

lightweight real-time ECG models

external dataset validation

ECG + PPG multimodal learning

self-supervised ECG representation learning

The correct final topic should be selected only after checking recent literature.

PDF limitation

This version works best with text-based PDFs. Scanned image-only PDFs may produce little extracted text.

For very long papers, the current version analyzes a limited amount of extracted text to avoid exceeding model context limits. A future version can add section-aware retrieval/RAG so the assistant searches the entire paper.

Safety

This application is for academic research assistance. It does not diagnose patients or provide treatment advice
