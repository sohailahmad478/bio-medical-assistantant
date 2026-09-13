# Biomedical Signal Research Gap Assistant

A Streamlit application for finding recent research gaps in biomedical signal processing.

## Features

- ECG, EEG, PPG, EMG, EOG, PCG, SCG, BioZ and multimodal signals
- Searches recent PubMed literature
- Collects paper metadata and abstracts
- Uses Groq to synthesize evidence-supported research gaps
- Ranks gaps for an MS project
- Suggests a final research title, datasets, metrics and roadmap
- Can be deployed on GitHub + Streamlit Community Cloud

## Files

```text
app.py
requirements.txt
readme.md
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud

Upload the three files to GitHub, create a Streamlit app, and select `app.py`.

Add this to Streamlit Secrets:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

## Recommended MS roadmap

1. Select one signal: ECG, EEG, PPG or EMG.
2. Search and read recent papers.
3. Identify repeated limitations.
4. Confirm the gap in original papers.
5. Select a public dataset.
6. Build a classical ML baseline.
7. Build a deep-learning baseline such as 1D-CNN/CNN-LSTM.
8. Add the proposed novelty: robustness, explainability, lightweight AI, self-supervised learning, multimodal fusion, or cross-subject validation.
9. Evaluate with F1, AUROC, sensitivity, specificity, calibration and robustness where appropriate.
10. Build a Streamlit demonstration.
11. Perform statistical/error analysis.
12. Write thesis and paper.

## Strong directions to investigate

### A. Robust ECG/PPG under signal degradation
Study motion artifact, poor sensor contact and missing/corrupted segments.

### B. Lightweight and explainable biomedical AI
Study whether a compact model can retain performance while providing interpretable predictions.

### C. Self-supervised biomedical signal learning
Use unlabeled ECG/PPG/EEG recordings for representation learning before supervised classification.

### D. Cross-subject/cross-device generalization
Test whether models trained on one group/device generalize to unseen subjects/devices.

### E. Multimodal ECG + PPG
Study adaptive fusion when one signal is noisy or unavailable.

These are candidate directions, not claims that no previous work exists. The original papers must be checked before finalizing a thesis gap.

## Example project direction

**Explainable and Lightweight Deep Learning for Robust ECG/PPG Analysis Under Signal Degradation**

Pipeline:

```text
Public ECG/PPG dataset
        ↓
Preprocessing
        ↓
Signal-quality assessment
        ↓
Noise/motion degradation
        ↓
1D-CNN or CNN-LSTM baseline
        ↓
Lightweight model
        ↓
Explainability
        ↓
Cross-subject testing
        ↓
Robustness analysis
        ↓
Streamlit demo
        ↓
Paper + thesis
```

## Safety

This is a research assistant, not a clinical diagnostic system. Verify all research claims against the original scientific papers.
