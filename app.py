import os
import re
import hashlib
import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(
    page_title="Biomedical Paper Gap Assistant",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 Biomedical Paper Gap Assistant")
st.caption(
    "RAG-based research assistant for finding possible future research gaps "
    "from uploaded biomedical signal-processing papers."
)

# -----------------------------
# Configuration
# -----------------------------
DEFAULT_MODEL = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SYSTEM_PROMPT = """
You are a biomedical signal-processing research assistant helping an MS student
identify possible research opportunities from a scientific paper.

IMPORTANT:
1. Separate what the authors explicitly state from what you infer.
2. Never claim that an inferred gap is definitely novel or unexplored.
3. Do not invent datasets, results, limitations, experiments, or citations.
4. Base your analysis primarily on the retrieved paper excerpts.
5. If the retrieved excerpts are insufficient, clearly say so.
6. This is a research-planning tool, not a medical diagnostic system.

The user wants practical MS-level research directions, especially for biomedical
signals such as ECG, EEG, EMG, PPG, EOG, PCG, and related physiological signals.

When asked for future gaps, use this structure:

1. Evidence from paper
2. Author-stated limitation/future work, if present
3. Possible research gap inferred from the evidence
4. Why the gap matters
5. Proposed improvement
6. Expected novelty: Low / Medium / High (as a cautious estimate)
7. Difficulty: Easy / Medium / Hard
8. MS suitability: Low / Medium / High

For the final recommendation, give:
- 3 possible MS research topics
- a recommended topic
- a cautious novelty statement
- possible public dataset directions
- a step-by-step research roadmap
- possible deployment/demo idea

Do not fabricate exact dataset names unless they are present in the paper or you
are explicitly asked for general suggestions. If giving general suggestions,
label them as suggestions rather than facts from the paper.
"""

# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def get_api_key():
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "")


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pdf_text(uploaded_file):
    data = uploaded_file.getvalue()
    doc = fitz.open(stream=data, filetype="pdf")

    pages = []
    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text")
        text = clean_text(text)
        if text:
            pages.append({
                "page": page_number,
                "text": text
            })

    return pages, len(doc)


def split_page_text(pages, chunk_size=1800, overlap=250):
    """
    Small chunks keep retrieved context safely below Groq TPM limits.
    Page number is retained as metadata.
    """
    chunks = []

    for item in pages:
        text = item["text"]
        page = item["page"]

        if len(text) <= chunk_size:
            chunks.append({
                "text": text,
                "page": page
            })
            continue

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()

            if chunk:
                chunks.append({
                    "text": chunk,
                    "page": page
                })

            if end >= len(text):
                break

            start = max(0, end - overlap)

    return chunks


def build_faiss_index(chunks, model):
    texts = [c["text"] for c in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings


def retrieve_chunks(query, chunks, index, model, top_k=5):
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    k = min(top_k, len(chunks))
    scores, indices = index.search(query_embedding, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        results.append({
            "text": chunks[idx]["text"],
            "page": chunks[idx]["page"],
            "score": float(score),
        })

    return results


def make_context(results):
    blocks = []

    for i, result in enumerate(results, start=1):
        blocks.append(
            f"[Retrieved Excerpt {i} | PDF page {result['page']}]\n"
            f"{result['text']}"
        )

    return "\n\n".join(blocks)


def ask_groq(client, model_name, question, context):
    user_prompt = f"""
You are analyzing an uploaded biomedical research paper.

USER QUESTION:
{question}

RETRIEVED PAPER EXCERPTS:
{context}

Answer using the retrieved excerpts as the main evidence.

For claims about the paper, mention PDF page numbers when possible.

If discussing a possible research gap, explicitly label it as:
"AI-inferred possible gap"

If the authors explicitly mention a limitation or future work, label it:
"Author-stated limitation/future work"

Do not present an AI inference as something the authors said.

For the question about future research gaps, prioritize practical MS-level
opportunities involving biomedical signal processing, robustness, explainability,
lightweight/edge deployment, multimodal signals, limited labels, cross-subject
generalization, noise/artifact handling, or other issues only when supported
by the retrieved evidence.

Keep the response structured and practical.
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=2200,
    )

    return response.choices[0].message.content


def analyze_target_sections(chunks, index, model, client, model_name):
    questions = {
        "Paper Summary": (
            "What is the research problem, signal type, dataset, methodology, "
            "and main result described in this paper?"
        ),
        "Limitations and Future Work": (
            "What limitations and future work do the authors explicitly state? "
            "Separate author-stated points from anything inferred."
        ),
        "Possible Research Gaps": (
            "Based on the paper evidence, what are the most useful possible "
            "future research gaps for an MS project? Give evidence, proposed "
            "improvement, novelty estimate, difficulty, and MS suitability."
        ),
        "MS Research Roadmap": (
            "Based on the paper evidence, propose 3 MS research topics, choose "
            "one recommended topic, explain the possible gap cautiously, suggest "
            "general dataset directions, and provide an 8-12 step roadmap."
        ),
    }

    outputs = {}

    for title, question in questions.items():
        results = retrieve_chunks(
            question,
            chunks,
            index,
            model,
            top_k=5,
        )
        context = make_context(results)
        outputs[title] = ask_groq(
            client,
            model_name,
            question,
            context,
        )

    return outputs


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    api_key_input = st.text_input(
        "Groq API Key",
        type="password",
        help="For deployment, preferably store GROQ_API_KEY in Streamlit Secrets.",
    )

    model_name = st.selectbox(
        "Groq Model",
        [
            "openai/gpt-oss-20b",
            "llama-3.3-70b-versatile",
        ],
        index=0,
    )

    top_k = st.slider(
        "Retrieved chunks per question",
        min_value=3,
        max_value=6,
        value=5,
    )

    st.markdown("---")
    st.info(
        "This version uses RAG: only the most relevant paper chunks are sent "
        "to Groq, reducing token usage and avoiding the previous 413 error."
    )

# -----------------------------
# Main UI
# -----------------------------
uploaded_file = st.file_uploader(
    "📄 Upload a biomedical research paper (PDF)",
    type=["pdf"],
)

preferred_topic = st.text_input(
    "Optional: preferred research direction",
    placeholder="e.g., ECG, EEG, PPG, wearable sensors, explainable AI",
)

if uploaded_file is None:
    st.markdown(
        """
### How it works

1. Upload one research paper in PDF format.
2. The app extracts the text with PyMuPDF.
3. The paper is divided into small chunks.
4. Sentence Transformers creates embeddings.
5. FAISS retrieves the most relevant chunks.
6. Groq analyzes only those chunks.
7. The app produces possible future research gaps and an MS roadmap.

**Important:** Author-stated future work and AI-inferred possible gaps are kept separate.
"""
    )
    st.stop()

# API key
api_key = api_key_input.strip() or get_api_key()

if not api_key:
    st.warning(
        "Please enter your Groq API key in the sidebar, or add GROQ_API_KEY "
        "to Streamlit Secrets."
    )
    st.stop()

# Extract
try:
    with st.spinner("📖 Extracting PDF text..."):
        pages, page_count = extract_pdf_text(uploaded_file)

    if not pages:
        st.error(
            "No selectable text was found. This may be a scanned/image-only PDF. "
            "Please use a text-based PDF or add OCR in a future version."
        )
        st.stop()

    full_text = " ".join(p["text"] for p in pages)

    if len(full_text) < 500:
        st.warning(
            "Very little text was extracted. The PDF may be scanned or have "
            "unusual formatting."
        )

    word_count = len(full_text.split())
    file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()

    c1, c2, c3 = st.columns(3)
    c1.metric("PDF pages", page_count)
    c2.metric("Extracted words", f"{word_count:,}")
    c3.metric("File size", f"{len(uploaded_file.getvalue()) / 1024:.1f} KB")

    with st.expander("🔎 Preview extracted text"):
        st.write(full_text[:5000])

    # Chunk
    with st.spinner("🧩 Splitting paper into RAG chunks..."):
        chunks = split_page_text(pages, chunk_size=1800, overlap=250)

    st.success(f"Created {len(chunks)} searchable chunks.")

    # Embeddings / FAISS
    with st.spinner("🧠 Building FAISS vector index..."):
        embedding_model = load_embedding_model()
        index, _ = build_faiss_index(chunks, embedding_model)

    st.success("FAISS index ready.")

    question = st.text_area(
        "🔬 What do you want to investigate?",
        value=(
            "Find the possible future research gaps in this paper and identify "
            "strong MS-level research opportunities."
        ),
        height=110,
    )

    if preferred_topic.strip():
        question += (
            f"\nThe student is particularly interested in: "
            f"{preferred_topic.strip()}."
        )

    if st.button("🚀 Analyze Research Gaps", type="primary", use_container_width=True):
        try:
            client = Groq(api_key=api_key)

            # Use the user's selected top_k for a focused one-shot answer first.
            retrieved = retrieve_chunks(
                question,
                chunks,
                index,
                embedding_model,
                top_k=top_k,
            )

            context = make_context(retrieved)

            with st.spinner("🤖 Groq is analyzing the retrieved evidence..."):
                answer = ask_groq(
                    client,
                    model_name,
                    question,
                    context,
                )

            st.markdown("---")
            st.subheader("🎯 Research Gap Analysis")
            st.markdown(answer)

            with st.expander("📚 Retrieved evidence"):
                for i, item in enumerate(retrieved, start=1):
                    st.markdown(
                        f"**Excerpt {i} — PDF page {item['page']} "
                        f"(similarity: {item['score']:.3f})**"
                    )
                    st.write(item["text"])

            st.download_button(
                "⬇️ Download Analysis as TXT",
                data=answer,
                file_name="biomedical_research_gap_analysis.txt",
                mime="text/plain",
            )

        except Exception as e:
            error_text = str(e)

            if "413" in error_text or "TPM" in error_text or "tokens per minute" in error_text:
                st.error(
                    "Groq rejected the request because the token-per-minute limit "
                    "was exceeded. Try fewer retrieved chunks (3) or use a smaller "
                    "question. The app already uses RAG to keep requests small."
                )
            else:
                st.error(f"Groq analysis failed: {error_text}")

except Exception as e:
    st.error(f"PDF/RAG processing failed: {e}")
