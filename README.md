---
title: Bike Demand Intelligence
emoji: 🚲
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Bike Demand Intelligence – Operations Assistant

Sigmoid Analytics case study (Capital Bikeshare, Washington DC, 2011–2012). A Streamlit assistant that answers plain-English questions about hourly bike demand using a gradient-boosting model trained on the UCI Bike Sharing dataset.

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Contents

| Path | What it is |
| --- | --- |
| `app.py` | Streamlit app |
| `src/` | Cleaning, features, model training, query engine |
| `models/best_model.joblib` | Trained model bundle (scikit-learn 1.7.2) |
| `data/` | `hour.csv`, `day.csv` (raw, unmodified) |
| `01`–`04` notebooks | Phase 1 data checks, Phase 2 analysis, Phase 3 modelling, Phase 4 prototype |
| `Dockerfile` | Container used for the Hugging Face Space |
