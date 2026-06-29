# 🏥 HealthAI — Multi-Disease Prediction System

## Overview
AI-powered simultaneous disease screening for:
- 🩸 **Diabetes** — Random Forest (AUC: 0.896)
- ❤️ **Heart Disease** — Gradient Boosting (AUC: 0.856)
- 🫁 **Liver Disease** — Random Forest (AUC: 0.797)
- 🫘 **Kidney Failure** — Gradient Boosting (AUC: 0.833)

## Features
- ✅ Multi-disease simultaneous prediction
- ✅ SHAP explainability for every prediction
- ✅ Downloadable PDF health report
- ✅ Doctor appointment booking interface
- ✅ Risk gauges + radar chart visualization
- ✅ Mobile-friendly dark UI

## Project Structure
```
health_ai/
├── app.py                  # Main Streamlit app
├── train_models.py         # Model training script
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Streamlit theme config
├── models/                 # Trained ML models (auto-generated)
│   ├── diabetes_model.pkl
│   ├── heart_model.pkl
│   ├── liver_model.pkl
│   └── kidney_model.pkl
└── utils/
    └── pdf_generator.py    # PDF report builder
```

## Local Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models (first time only)
python train_models.py

# 3. Run the app
streamlit run app.py
```

## Deploy to Streamlit Cloud (Free)
1. Push this folder to a **GitHub repo**
2. Go to → https://share.streamlit.io
3. Connect your GitHub, select the repo
4. Set **Main file:** `app.py`
5. Click **Deploy** — your public URL is ready in ~2 minutes

## Deploy to Render (Free Tier)
1. Push to GitHub
2. Go to https://render.com → New Web Service
3. Connect repo, set:
   - **Build Command:** `pip install -r requirements.txt && python train_models.py`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

## Target Users
- 💊 Health-conscious individuals
- 🏥 Small clinics (pre-screening)
- 🏢 Corporate wellness programs

## Disclaimer
For educational use only. Not a substitute for professional medical advice.
