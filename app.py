import os
import re
import streamlit as st
import fitz  # PyMuPDF
from groq import Groq

st.set_page_config(
    page_title="Biomedical Paper Gap Assistant",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Biomedical Research Gap Assistant")
st.write(
    "Upload a biomedical signal-processing research paper and get "
    "evidence-based possible future research gaps, MS-level topics, and a roadmap."
)

st.warning(
    "Research support only. The suggested gaps are hypotheses/inferences, not proof of novelty. "
    "Always verify them against the original literature."
)

# ---------- API KEY ----------
def get_api_key():
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        key = ""
    return key or os.getenv("GROQ_API_KEY", "")

with st.sidebar:
    st.header("⚙️ Settings")

    saved_key = get_api_key()
    api_key = st.text_input(
        "Groq API Key",
        value=saved_key,
        type="password",
        help="For Streamlit Cloud, store GROQ_API_KEY in App Settings > Secrets."
    )

    model = st.selectbox(
        "Groq model",
        ["openai/gpt-oss-20b", "llama-3.3-70b-versatile"],
        index=0
    )

    response_style = st.selectbox(
        "Analysis depth",
        ["MS Project Focused", "Detailed Research Analysis"],
        index=0
    )

# ---------- PDF EXTRACTION ----------
def extract_pdf_text(uploaded_file):
    pdf_bytes = uploaded_file.getvalue()

    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    pages = []
    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text")
        text = re.sub(r"\s+", " ", text).strip()

        if text:
            pages.append(
                f"\n--- PAGE {page_number} ---\n{text}"
            )

    doc.close()

    full_text = "\n".join(pages)

    if len(full_text.strip()) < 500:
        raise ValueError(
            "Very little text was extracted. This may be a scanned/image-only PDF. "
            "Please upload a text-based PDF."
        )

    return full_text

# ---------- TEXT CHUNKING ----------
def split_text(text, max_chars=12000):
    words = text.split()
    chunks = []
    current = []

    current_len = 0

    for word in words:
        if current_len + len(word) + 1 > max_chars and current:
            chunks.append(" ".join(current))
            current = []
            current_len = 0

        current.append(word)
        current_len += len(word) + 1

    if current:
        chunks.append(" ".join(current))

    return chunks

# ---------- GROQ ANALYSIS ----------
def analyze_paper(client, model, paper_text, depth):
    chunks = split_text(paper_text, 12000)

    # Avoid exceeding Groq token limits.
    max_chunks = 5
    selected_chunks = chunks[:max_chunks]

    context = "\n\n".join(
        f"===== PAPER SECTION {i + 1} =====\n{chunk}"
        for i, chunk in enumerate(selected_chunks)
    )

    system_prompt = """You are an expert biomedical signal-processing research advisor.

You help an MS student identify FUTURE RESEARCH POSSIBILITIES from an uploaded research paper.

Common biomedical signals include ECG, EEG, EMG, PPG, EOG, PCG, SCG and multimodal physiological signals.

IMPORTANT RULES:
1. First identify what the paper actually did.
2. Identify limitations explicitly stated by the authors.
3. Identify future work explicitly stated by the authors.
4. Then generate additional POSSIBLE research gaps by reasoning from the paper.
5. Clearly label author-stated limitations/future work versus AI-inferred possibilities.
6. Never claim that an inferred gap is completely unexplored.
7. Do not invent datasets, numerical results, or claims that are not supported by the paper.
8. Explain why each proposed gap could be useful.
9. Prefer realistic MS-level research directions.
10. Do not provide medical diagnosis or treatment advice.

Potential gap dimensions to examine:
- small dataset
- class imbalance
- limited subjects
- single-center/single-device data
- cross-subject generalization
- cross-device/domain generalization
- noise and motion artifacts
- signal quality
- missing data
- missing modalities
- real-time performance
- computational cost
- lightweight/edge deployment
- explainability
- uncertainty/calibration
- robustness
- self-supervised learning
- transfer learning
- multimodal fusion
- privacy/federated learning
- external validation
- reproducibility
- clinical validation
"""

    if depth == "MS Project Focused":
        output_request = """Give a practical MS-focused report."""
    else:
        output_request = """Give a detailed research-analysis report."""

    user_prompt = f"""Analyze the following biomedical research paper.

{output_request}

Return these sections:

## 1. Paper Identification
- likely title
- biomedical signal(s)
- research problem
- main objective

## 2. What the Authors Actually Did
- dataset
- preprocessing
- method/model
- evaluation
- main result

## 3. Author-Stated Limitations
Only list limitations that are supported by the paper.

## 4. Author-Stated Future Work
Only list future work actually mentioned in the paper.
If it is not available in the supplied text, say "Not clearly stated in the supplied text."

## 5. Possible Future Research Gaps
Give 6-10 possibilities.

For EVERY gap use:
- Gap
- Evidence from this paper
- Why it matters
- Proposed improvement
- Expected novelty: Low / Medium / High
- Difficulty: Easy / Medium / Hard
- MS suitability: Low / Medium / High

## 6. Top 3 MS Research Opportunities
Rank them and explain the ranking.

## 7. Recommended MS Research Topic
Give ONE strong title based on the paper.

## 8. Novelty Statement
Give a cautious 2-3 sentence novelty statement.

## 9. Dataset Suggestions
Suggest appropriate public biomedical signal datasets, but clearly label them as suggestions rather than claims from the paper.

## 10. Experimental Roadmap
Give 8-12 steps from dataset selection to thesis/paper.

## 11. Possible Deployment
Explain whether the proposed work could reasonably be demonstrated using Streamlit.

## 12. Important Warning
Explain what must be verified through additional literature before claiming novelty.

PAPER TEXT:
{context}
"""

    response = client.chat.completions.create(
        model=model,
        temperature=0.15,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content, len(chunks), len(selected_chunks)

# ---------- UI ----------
uploaded_file = st.file_uploader(
    "📄 Upload your biomedical research paper (PDF)",
    type=["pdf"],
    accept_multiple_files=False
)

topic = st.text_input(
    "Optional: What area are you interested in?",
    placeholder="Example: ECG, PPG, EEG, explainable AI, lightweight deep learning..."
)

if uploaded_file is not None:
    st.success(f"Uploaded: {uploaded_file.name}")

    try:
        paper_text = extract_pdf_text(uploaded_file)

        word_count = len(paper_text.split())
        page_count = paper_text.count("--- PAGE ")

        c1, c2, c3 = st.columns(3)
        c1.metric("Pages detected", page_count)
        c2.metric("Words extracted", f"{word_count:,}")
        c3.metric(
            "File size",
            f"{len(uploaded_file.getvalue()) / 1024 / 1024:.2f} MB"
        )

        with st.expander("👁️ Preview extracted paper text"):
            preview = paper_text[:6000]
            st.write(preview + ("..." if len(paper_text) > 6000 else ""))

        if st.button("🔎 Analyze Future Research Gaps", type="primary"):
            if not api_key:
                st.error(
                    "Groq API key is missing. Add GROQ_API_KEY in Streamlit "
                    "Secrets or enter it in the sidebar."
                )
                st.stop()

            client = Groq(api_key=api_key)

            if topic.strip():
                analysis_input = (
                    paper_text
                    + "\n\nSTUDENT'S PREFERRED RESEARCH AREA:\n"
                    + topic.strip()
                )
            else:
                analysis_input = paper_text

            with st.spinner(
                "Reading the paper and identifying future research gaps..."
            ):
                try:
                    report, total_chunks, used_chunks = analyze_paper(
                        client=client,
                        model=model,
                        paper_text=analysis_input,
                        depth=response_style
                    )
                except Exception as e:
                    st.error(f"Groq analysis failed: {e}")
                    st.stop()

            st.markdown("---")
            st.header("🎯 Future Research Gap Analysis")
            st.markdown(report)

            st.download_button(
                "⬇️ Download Analysis as TXT",
                data=report,
                file_name="biomedical_research_gap_analysis.txt",
                mime="text/plain"
            )

            with st.expander("ℹ️ Processing information"):
                st.write(f"Text chunks detected: {total_chunks}")
                st.write(f"Chunks analyzed: {used_chunks}")
                st.write(
                    "For very long papers, the current version analyzes a limited "
                    "number of chunks to stay within the model context limit."
                )

    except Exception as e:
        st.error(f"Could not read the PDF: {e}")

else:
    st.markdown(
        """
### How to use

1. Upload a biomedical signal-processing research paper.
2. Optionally enter your preferred area such as **ECG, PPG, EEG, EMG,
   explainable AI, lightweight deep learning, or wearable monitoring**.
3. Click **Analyze Future Research Gaps**.
4. The assistant separates:
   - what the authors actually did
   - author-stated limitations
   - author-stated future work
   - AI-inferred possible research gaps
   - top MS opportunities
   - a proposed research title
   - dataset suggestions
   - an implementation roadmap

### Example

Upload a paper about:

**Deep Learning-Based ECG Arrhythmia Classification**

The assistant can investigate possible directions such as:

- cross-subject validation
- robustness to noisy ECG
- explainable ECG classification
- lightweight models for edge devices
- external dataset validation
- multimodal ECG + PPG fusion

These are **possible research directions**, not automatic proof of novelty.
"""
    )

st.markdown("---")
st.caption(
    "Biomedical research assistant — not a clinical diagnostic system. "
    "Always verify proposed gaps using the original paper and additional recent literature."
)
