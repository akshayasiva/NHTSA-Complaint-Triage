import json, joblib
from fastapi import FastAPI
from pydantic import BaseModel

comp_lr   = joblib.load("models/comp_lr.joblib")
danger_lr = joblib.load("models/danger_lr.joblib")
cfg       = json.load(open("models/config.json"))
id2label  = {int(k): v for k, v in cfg["id2label"].items()}
THRESHOLD = cfg["danger_threshold"]

app = FastAPI(title="NHTSA Complaint Triage")

class Complaint(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(c: Complaint):
    text = c.text.lower()
    probs = comp_lr.predict_proba([text])[0]
    top3 = probs.argsort()[::-1][:3]
    danger = float(danger_lr.predict_proba([text])[0, 1])
    return {
        "component": id2label[int(comp_lr.classes_[top3[0]])],
        "top3": [{"component": id2label[int(comp_lr.classes_[i])], "prob": round(float(probs[i]), 3)} for i in top3],
        "danger_score": round(danger, 3),
        "danger_flag": danger >= THRESHOLD,
    }
