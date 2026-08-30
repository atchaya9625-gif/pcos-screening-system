# PCOS Detection + Personalized Lifestyle Recommendation System

## Structure
- app/            -> Flask web app (routes, templates, static files)
- data/raw/        -> original downloaded datasets (put your 5 datasets here)
- data/processed/  -> cleaned datasets after preprocessing
- models/          -> saved trained models (.pkl, .h5, etc.)
- notebooks/       -> Jupyter notebooks for EDA + model training/experiments

## How to run
1. pip install -r requirements.txt
2. python run.py
3. Open http://127.0.0.1:5000

## Pipeline
Stage 1: Detection (ultrasound + clinical fusion model) -> /predict
Stage 2: Explainability (SHAP) -> feeds into predict route response
Stage 3: Personalized recommendation + nudges -> /dashboard
