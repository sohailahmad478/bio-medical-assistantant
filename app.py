import os
import re
import requests
import streamlit as st
from groq import Groq

st.set_page_config(page_title="Biomedical Signal Research Gap Assistant", page_icon="🧬", layout="wide")

st.title("🧬 Biomedical Signal Research Gap Assistant")
st.caption("Find recent research gaps in ECG, EEG, EMG, PPG and other biomedical signals.")

st.info("Research-planning tool only. Verify every proposed gap against the original papers before using it in an MS thesis or publication.")

with st.sidebar:
    st.header("Settings")
    secret_key = st.secrets.get("GROQ_API_KEY", "")
    api_key = st.text_input("Groq API Key", value=secret_key, type="password")
    model = st.selectbox("Groq model", ["openai/gpt-oss-20b", "llama-3.3-70b-versatile"])
    years = st.slider("Recent literature window (years)", 1, 10, 3)
    n_papers = st.slider("PubMed papers", 5, 20, 10)

def pubmed_search(query, max_results=10):
    params = {"db":"pubmed","term":query,"retmode":"json","retmax":max_results,"sort":"pub date"}
    r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params=params, timeout=30)
    r.raise_for_status()
    ids = r.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []

    s = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
        params={"db":"pubmed","id":",".join(ids),"retmode":"json"},
        timeout=30
    )
    s.raise_for_status()
    summaries = s.json().get("result", {})

    f = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        params={"db":"pubmed","id":",".join(ids),"retmode":"xml"},
        timeout=30
    )
    f.raise_for_status()

    abstracts = {}
    for block in re.findall(r"<PubmedArticle>(.*?)</PubmedArticle>", f.text, flags=re.DOTALL):
        pm = re.search(r"<PMID[^>]*>(.*?)</PMID>", block)
        if not pm:
            continue
        texts = re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>", block, flags=re.DOTALL)
        abstract = " ".join(re.sub(r"<.*?>", " ", x) for x in texts)
        abstracts[pm.group(1)] = re.sub(r"\s+", " ", abstract).strip()

    papers = []
    for pid in ids:
        item = summaries.get(pid, {})
        papers.append({
            "pmid": pid,
            "title": item.get("title", ""),
            "journal": item.get("fulljournalname", ""),
            "date": item.get("pubdate", ""),
            "authors": ", ".join(a.get("name", "") for a in item.get("authors", [])[:5]),
            "abstract": abstracts.get(pid, "")
        })
    return papers

def literature_text(papers):
    return "\n\n".join(
        f"[Paper {i}]\nPMID: {p['pmid']}\nTitle: {p['title']}\n"
        f"Journal: {p['journal']}\nDate: {p['date']}\n"
        f"Authors: {p['authors']}\nAbstract: {p['abstract'][:5000]}"
        for i, p in enumerate(papers, 1)
    )

def analyze_gaps(client, model, topic, signal, notes, literature):
    system = """You are a biomedical signal-processing research advisor for an MS student.
Focus on ECG, EEG, EMG, PPG, EOG, PCG, SCG, BioZ and related physiological signals.

Do not invent a research gap. Separate evidence-supported observations from your inference.
Consider dataset diversity, class imbalance, cross-subject generalization, noise/motion
artifacts, missing modalities, domain shift, explainability, uncertainty, lightweight/edge
deployment, real-time performance, self-supervised learning, multimodal fusion, privacy,
robustness, signal quality and clinical validation.

Do not claim a topic is completely unexplored unless the supplied evidence clearly supports it.
Do not give patient diagnosis or treatment advice."""

    prompt = f"""Topic: {topic}
Primary signal: {signal}
Student notes: {notes or "None"}

Recent PubMed literature:
{literature}

Prepare an MS research-gap report:

1. Current research direction
2. 5-8 research gaps. For each: gap, evidence, why it matters, difficulty, experiment.
3. Rank the top 3 gaps for an 8-12 month MS project.
4. Recommend ONE final research title.
5. Give a 2-3 sentence novelty statement.
6. Suggest datasets and evaluation metrics.
7. Give a step-by-step roadmap from dataset to paper and Streamlit demo.
8. List risks and limitations.

Keep it practical and beginner-friendly."""
    result = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[{"role":"system","content":system},{"role":"user","content":prompt}]
    )
    return result.choices[0].message.content

topic = st.text_input("Biomedical signal research topic", "Deep learning for ECG and PPG signal analysis")
signal = st.selectbox("Primary signal", ["ECG","EEG","PPG","EMG","EOG","PCG","SCG","BioZ","Multimodal"])
notes = st.text_area(
    "Optional: paste abstracts, supervisor notes, or your current idea",
    height=150,
    placeholder="Example: I want an MS project that can be completed in 8-12 months and later demonstrated in Streamlit."
)

if st.button("🔎 Find Recent Research Gaps", type="primary"):
    if not api_key:
        st.error("Add your Groq API key in Streamlit Secrets or the sidebar.")
        st.stop()

    query = (
        f'({signal}[Title/Abstract] OR "biomedical signal"[Title/Abstract]) '
        f'AND ("deep learning"[Title/Abstract] OR "machine learning"[Title/Abstract] '
        f'OR transformer[Title/Abstract] OR "self-supervised"[Title/Abstract]) '
        f'AND ("{years} years"[PDat] : "3000"[PDat])'
    )

    with st.spinner("Searching recent PubMed literature..."):
        try:
            papers = pubmed_search(query, n_papers)
        except Exception as e:
            st.error(f"PubMed search failed: {e}")
            st.stop()

    if not papers:
        st.warning("No papers found. Try a broader topic.")
        st.stop()

    st.success(f"Retrieved {len(papers)} recent PubMed records.")

    with st.expander("📚 Literature retrieved"):
        for p in papers:
            st.markdown(f"**{p['title']}**")
            st.caption(f"PMID {p['pmid']} | {p['date']} | {p['journal']}")
            if p["abstract"]:
                st.write(p["abstract"][:1200] + ("..." if len(p["abstract"]) > 1200 else ""))

    with st.spinner("Analyzing research gaps with Groq..."):
        try:
            client = Groq(api_key=api_key)
            report = analyze_gaps(client, model, topic, signal, notes, literature_text(papers))
        except Exception as e:
            st.error(f"Groq request failed: {e}")
            st.stop()

    st.markdown("---")
    st.header("🎯 Research Gap Report")
    st.markdown(report)

    st.markdown("---")
    st.subheader("🔗 Literature used")
    for p in papers:
        st.markdown(f"- **{p['title']}** — PMID {p['pmid']} ([PubMed](https://pubmed.ncbi.nlm.nih.gov/{p['pmid']}/))")

st.markdown("---")
st.caption("Always read and verify the original papers. This tool is not a medical diagnostic system.")
