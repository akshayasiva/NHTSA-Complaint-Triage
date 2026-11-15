

# Vehicle Safety Complaint Triage (NHTSA)

An NLP system that reads real vehicle safety complaints filed with the US government (NHTSA), **classifies the failed component**, **flags dangerous complaints** (crash, fire, injury, death), **finds similar past complaints**, and **writes a monthly emerging-issues report** for engineers.

**Research question:** Is a fine-tuned DistilBERT worth its extra cost over a simple TF-IDF + logistic regression model?

[Triage app]<img width="1258" height="505" alt="SS_2" src="https://github.com/user-attachments/assets/8c70ecfb-da0f-4412-bd8a-615195e67fd7" />

---

## Why this matters

NHTSA receives tens of thousands of owner complaints a year. Safety engineers can't read them all, yet a cluster of similar complaints is often the first sign of a defect that leads to a recall. This project triages each complaint the way a nurse triages patients: what's wrong, and how urgent is it.

---

## Results

Test set: **15,000 complaints received in 2024**. Models trained on 2021–2023.

| Model | Macro-F1 (component) | Danger PR-AUC | Danger recall | Danger precision | ms / prediction |
|---|---|---|---|---|---|
| Dumb baseline | 0.038 | 0.041 | 0.000 | 0.000 | 0 |
| TF-IDF + LogReg | 0.885 | 0.830 | 0.763 | 0.796 | 0.34 (CPU) |
| DistilBERT (fine-tuned) | **0.897** | **0.876** | **0.781** | **0.906** | 3.91 (GPU) |

- **Dumb baseline:** always predicts the most common component; random danger score
- **Danger threshold:** chosen on a validation set to target 80% recall (TF-IDF 0.56, DistilBERT 0.72)

### Drift check (2025–2026 complaints)

| Check | Value | Status |
|---|---|---|
| Component mix PSI | 0.083 | 🟢 Stable |
| Danger score PSI | 0.027 | 🟢 Stable |
| Text length PSI | 0.052 | 🟢 Stable |
| Macro-F1: test (2024) → future (2025–26) | 0.885 → 0.873 | Small drop |

PSI guide: < 0.10 stable, 0.10–0.25 watch, > 0.25 alert.

---

## Key findings

- **DistilBERT wins, but only slightly on component classification:** +1.2 points macro-F1, while being ~12× slower per prediction.
- **The bigger gain is on danger detection:** PR-AUC +4.6 points and precision 0.80 → 0.91, which means roughly half as many false alarms for engineers to review, at similar recall.
- **Deployment choice:** the API serves the TF-IDF model, which is cheap, CPU-only and sub-millisecond. DistilBERT is worth adding for the danger flag when a GPU is available.
- **Most errors are between related systems:** power train ↔ engine (281 cases), then electrical ↔ engine. Some "errors" are really label noise, such as a broken horn labeled STEERING (the horn sits on the steering wheel) or recall-notice complaints that never name a part.
- **Recall landed below the 80% target on test (0.76–0.78):** the threshold was tuned on 2021–23 data, and the danger rate fell from 6.5% in 2021 to 3.9% in 2024. This is exactly the kind of shift a random split would have hidden.
- **The model held up on newer data:** macro-F1 dropped only 0.012 on 2025–26 complaints, and no PSI alert fired.
- **The spike detector found real October 2026 clusters,** for example Mazda CX-90 steering (18 complaints vs 7.7 usual, 2.3×). The LLM summary cites the complaint IDs it used.
- **The word-counting model has blind spots:** TF-IDF scored a "rear-ended the car in front of me" complaint at 0.44 and missed it as a crash, because it counts words rather than understanding phrases.

---

## Pipeline

```
Download NHTSA flat files
  → clean + create labels (component, danger)
  → time-based split (train 2021–23 | test 2024 | future 2025–26)
  → TF-IDF + logistic regression baseline
  → fine-tuned DistilBERT (same data, same metrics)
  → error analysis + explainable top words
  → MLflow experiment tracking
  → similar-complaint search (sentence embeddings + Chroma)
  → spike detection + grounded LLM report (Qwen2.5-1.5B)
  → PSI drift monitoring
  → FastAPI service + Docker
  → Gradio demo app
```

---

## Data

- **Source:** NHTSA Office of Defects Investigation (ODI) complaints flat files, 2020–2026
- **Raw:** 633,275 rows → **247,376 clean complaints**
- **Cleaning:**
  - Vehicles only (dropped tires, child seats, equipment)
  - Complaints naming exactly one main component, one row per complaint
  - Dropped "unknown" labels and complaints under 20 words
  - **Privacy:** dropped owner names, VINs and dealer contact details
- **Labels (from NHTSA, no manual labeling):**
  - **Component:** main system, top 10 (engine, power train, electrical, steering, air bags, service brakes, structure, fuel/propulsion, forward collision avoidance, visibility/wiper)
  - **Danger:** 1 if crash, fire, injury or death was reported; overall rate 5.2%
- **Split by time, not randomly:** train on the past, test on the future, like real deployment

| Set | Received | Size |
|---|---|---|
| Train | 2021–2023 | 54,000 |
| Validation | 2021–2023 (held out) | 6,000 |
| Test | 2024 | 15,000 |
| Future (drift) | 2025–2026 | 15,000 |

Train/test overlap check: **0** shared complaints.

---

## Demo app

Three tabs built with Gradio:

1. **Triage:** paste a complaint → top 3 components with confidence + danger flag
2. **Similar:** find the 5 most similar past complaints, optionally filtered by make
3. **Emerging issues:** this month's spikes with an LLM summary and evidence IDs

---

## API

```bash
docker build -t nhtsa-triage .
docker run -p 8000:8000 nhtsa-triage
```

Then open `http://localhost:8000/docs`.

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The airbag light stays on and the passenger airbag did not deploy in a crash"}'
```

Example response:

```json
{
  "component": "AIR BAGS",
  "top3": [{"component": "AIR BAGS", "prob": 1.0}, ...],
  "danger_score": 0.857,
  "danger_flag": true
}
```

---

## Run it yourself

1. Open `NHTSA_Complaint_Triage.ipynb` in Google Colab
2. **Runtime → Change runtime type → T4 GPU**
3. Run all cells top to bottom (about 60–90 minutes)

The notebook downloads the NHTSA data itself. Data and trained models are not stored in this repo because of their size.

---

## Tech stack

Python, pandas, scikit-learn, Hugging Face Transformers (DistilBERT), sentence-transformers, ChromaDB, Qwen2.5-1.5B-Instruct, MLflow, FastAPI, Docker, Gradio, Google Colab (T4 GPU)

---

## Limitations

- Only single-component complaints are used (~40% of rows); multi-part complaints are left out
- Only the top 10 components are classified
- NHTSA component names changed over time (e.g., `ENGINE` vs `ENGINE AND ENGINE COOLING`), so some older labels were left out
- Owner-written text is messy, and some labels are debatable (label noise)
- The danger label depends on owners ticking the crash/fire/injury boxes
- The speed comparison is not apples-to-apples: DistilBERT ran on a GPU, TF-IDF on a CPU
- Grounding reduces but doesn't remove LLM over-reach, so the report always shows evidence IDs for an engineer to check
- One random seed; US data only

---

## Next steps

- Multi-label classification to cover complaints that name several parts
- Re-tune the danger threshold on recent data to recover 80% recall
- Link detected spikes to actual NHTSA recalls and measure how early they'd have been caught
- Compare DistilBERT epochs (1 vs 2 vs 3) and input length (128 vs 256 vs 512 tokens)
- Deploy the demo permanently on Hugging Face Spaces

---

## Data source

National Highway Traffic Safety Administration, Office of Defects Investigation — Complaints flat files (public data).
