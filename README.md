# Vehicle Safety Complaint Triage (NHTSA)

Classifies the failed component and flags dangerous complaints (crash, fire, injury)
from real NHTSA vehicle owner complaints, then finds similar complaints and writes
an emerging-issues report.

**Question:** Is a fine-tuned DistilBERT worth its extra cost over TF-IDF + logistic regression?

<img width="1501" height="460" alt="SS_1" src="https://github.com/user-attachments/assets/cf0bb51d-a290-4a1e-85c8-6e2fb8243219" />

## Results
| Model | Macro-F1 | Danger PR-AUC | Danger recall | ms / prediction |
|---|---|---|---|---|
| Dumb baseline | ___ | ___ | ___ | 0 |
| TF-IDF + LogReg | ___ | ___ | ___ | ___ |
| DistilBERT | ___ | ___ | ___ | ___ |

## Pipeline
Download → clean → time-based split → TF-IDF baseline → DistilBERT → error analysis
→ MLflow → similar search (Chroma) → spike detection + LLM report → PSI drift → FastAPI + Docker → Gradio

## Data
NHTSA ODI complaints, 2020–2026. Train 2021–23, test 2024, drift check 2025–26.
Privacy fields (names, VINs, dealer info) dropped.

## Key findings
- Example: TF-IDF missed "rear-ended" as a crash (score 0.44); DistilBERT scored ___

## Run it
Open the notebook in Google Colab (T4 GPU) and run top to bottom.
API: `docker build -t nhtsa-triage . && docker run -p 8000:8000 nhtsa-triage`

## Limitations
Single-component complaints only, top 10 components, owner-written text with label noise, one seed.
