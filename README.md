# An Explainable Multimodal AI Framework for PCOS Screening with Personalized, Continuous Lifestyle Recommendation

A comprehensive clinical-AI system for Polycystic Ovary Syndrome (PCOS) screening that combines multimodal fusion (clinical biomarkers + pelvic ultrasound deep learning), transparent explainable AI (SHAP + Grad-CAM), and time-slotted personalized lifestyle recommendations.

---

## 🌟 Key Highlights

- **Multimodal AI Screening**: Fuses 12 Mutual-Information-selected clinical/hormonal biomarkers with pelvic ultrasound imaging via decision-level averaging.
- **Clinical Model**: XGBoost Classifier achieving **92.7% accuracy** and **0.969 AUC-ROC** on 899 merged patient records (73% feature dimensionality reduction from 45 down to 12 features).
- **Ultrasound Image Model**: EfficientNetB0 CNN with two-phase transfer learning (frozen warm-up + fine-tuning of top 30 layers), achieving **85.2% accuracy**, **90.1% precision**, **84.4% recall**, and **0.941 AUC-ROC** across ~3,850 ultrasound images.
- **Explainable AI (XAI)**:
  - **SHAP (SHapley Additive exPlanations)**: Dynamic waterfall/bar charts explaining clinical feature contributions per patient.
  - **Grad-CAM (Gradient-weighted Class Activation Mapping)**: Heatmap overlays highlighting regions of interest (follicular clusters) in ultrasound scans.
- **Phenotype-Tailored Lifestyle Engine**: Rule-based phenotype classification (*Insulin-Resistant*, *Inflammatory/Androgen-Related*, *Cycle-Irregularity Dominant*) delivering 4 time-slotted daily nudges (morning, afternoon, evening, night) with food and yoga/exercise recommendations via the browser Notifications API.
- **Full History Tracking**: SQLite database via Flask-SQLAlchemy tracking all predictions in India Standard Time (IST).

---

## 📂 Project Architecture

```
pcos_project/
├── app/
│   ├── __init__.py           # Flask app factory & database initialization
│   ├── routes.py             # Web routes, model loading, SHAP & Grad-CAM generation
│   ├── models.py             # SQLAlchemy PredictionHistory database schema
│   ├── recommendations.py    # Phenotype classification & lifestyle recommendation data
│   ├── templates/
│   │   ├── index.html        # Landing page with project highlights
│   │   ├── predict.html      # Multimodal screening input form & interactive results
│   │   ├── dashboard.html    # Personalized lifestyle dashboard with browser reminders
│   │   └── history.html      # Prediction history log
│   └── static/
│       ├── css/style.css     # Responsive, styled UI with accessible components
│       └── js/predict.js     # Form validation, preset demo auto-fill, image preview
├── models/                   # Trained production models & encoders
│   ├── pcos_clinical_model_v2.pkl  # Trained XGBoost classifier
│   ├── selected_features_v2.pkl    # 12 selected clinical features
│   └── pcos_image_model.keras      # EfficientNetB0 fine-tuned CNN
├── notebooks/                # Research and model training notebooks
│   ├── 01_clinical_model_training.ipynb
│   ├── 02_feature_selection_optimization.ipynb
│   ├── 03_image_model_training_4.ipynb
│   └── 04_image_model_finetuned.ipynb
├── requirements.txt          # Production dependencies (tensorflow-cpu, etc.)
├── render.yaml               # Render.com cloud deployment configuration
├── test_app.py               # Automated end-to-end integration test suite
└── run.py                    # Application entrypoint
```

---

## 🚀 Running Locally

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Git

### 2. Setup
```bash
git clone https://github.com/atchaya9625-gif/pcos-screening-system.git
cd pcos-screening-system
pip install -r requirements.txt
```

### 3. Run Automated Tests
```bash
python test_app.py
```

### 4. Start the Application
```bash
python run.py
```
Open your browser at `http://127.0.0.1:5000`.

---

## 🩺 Clinical Biomarkers (12 MI-Selected Features)

1. **Follicle Count (Right & Left Ovary)**
2. **Cycle Regularity & Cycle Length (Days)**
3. **Anti-Müllerian Hormone (AMH)**
4. **Prolactin (PRL)**
5. **FSH / LH Ratio**
6. **Clinical Symptoms**: Weight gain, Hirsutism (excess hair growth), Acanthosis Nigricans (skin darkening), Pimples/Acne, Fast food consumption

---

## 🔒 Medical Disclaimer

This system is an AI-assisted screening research prototype designed for academic, educational, and decision-support purposes. It does not replace clinical consultation, laboratory diagnostics, or medical advice by a qualified healthcare professional.
