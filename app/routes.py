import os
import io
import base64
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for Flask
import matplotlib.pyplot as plt
from flask import Blueprint, render_template, request, jsonify, session

from app.recommendations import classify_phenotype, get_recommendations
from app.models import db, PredictionHistory

main = Blueprint('main', __name__)

# ============================================================
# Load trained models + supporting files (once, at startup)
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLINICAL_MODEL_PATH = os.path.join(BASE_DIR, "models", "pcos_clinical_model_v2.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "selected_features_v2.pkl")
IMAGE_MODEL_PATH = os.path.join(BASE_DIR, "models", "pcos_image_model.keras")

IMG_SIZE = 160

clinical_model = None
selected_features = None
image_model = None
shap_explainer = None


def load_clinical_model():
    global clinical_model, selected_features
    if clinical_model is None and os.path.exists(CLINICAL_MODEL_PATH):
        clinical_model = joblib.load(CLINICAL_MODEL_PATH)
        selected_features = joblib.load(FEATURES_PATH)
    return clinical_model, selected_features


def load_image_model():
    global image_model
    if image_model is None and os.path.exists(IMAGE_MODEL_PATH):
        import tensorflow as tf
        image_model = tf.keras.models.load_model(IMAGE_MODEL_PATH)
    return image_model


def get_shap_explainer(mdl):
    global shap_explainer
    if shap_explainer is None:
        import shap
        shap_explainer = shap.TreeExplainer(mdl)
    return shap_explainer


def fig_to_base64(fig):
    """Convert a matplotlib figure to a base64 data URI string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def generate_shap_chart(mdl, X, features):
    """Generate a SHAP feature-contribution bar chart for a single prediction."""
    try:
        import shap
        explainer = get_shap_explainer(mdl)
        shap_values = explainer.shap_values(X)

        # shap_values shape: (1, n_features) for a single row
        values = shap_values[0] if shap_values.ndim == 2 else shap_values
        contributions = pd.Series(values, index=features).sort_values(key=abs)

        colors = ["#D6336C" if v > 0 else "#6C5CE7" for v in contributions.values]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(contributions.index, contributions.values, color=colors)
        ax.set_xlabel("Impact on PCOS risk (SHAP value)")
        ax.set_title("Why this prediction? (Clinical features)")
        ax.axvline(0, color="grey", linewidth=0.8)
        plt.tight_layout()

        return fig_to_base64(fig)
    except Exception as e:
        print(f"SHAP explanation generation error: {e}")
        return None


def make_gradcam_overlay(img_model, img_array_norm):
    """Generate a Grad-CAM heatmap overlay for the given preprocessed image."""
    try:
        import tensorflow as tf
        import cv2

        base_model = img_model.layers[0]
        last_conv_layer = base_model.get_layer("top_conv")
        conv_model = tf.keras.models.Model(base_model.input, last_conv_layer.output)
        classifier_layers = img_model.layers[1:]

        img_tensor = tf.convert_to_tensor(img_array_norm)

        with tf.GradientTape() as tape:
            conv_output = conv_model(img_tensor)
            tape.watch(conv_output)
            x = conv_output
            for layer in classifier_layers:
                x = layer(x)
            loss = x[:, 0]

        grads = tape.gradient(loss, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_output = conv_output[0]
        heatmap = conv_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0

        original = img_array_norm[0]
        overlay = 0.6 * original + 0.4 * heatmap_colored

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(original)
        axes[0].set_title("Ultrasound Image")
        axes[0].axis("off")
        axes[1].imshow(overlay)
        axes[1].set_title("Grad-CAM: Model Focus Area")
        axes[1].axis("off")
        plt.tight_layout()

        return fig_to_base64(fig)
    except Exception as e:
        print(f"Grad-CAM overlay error: {e}")
        return None


def predict_image_risk(file_storage):
    """
    Takes an uploaded image file, preprocesses it, and returns
    (risk_probability, gradcam_base64_image_or_None).
    Returns (None, None) if no image model/file is available.
    """
    img_model = load_image_model()
    if img_model is None or file_storage is None or file_storage.filename == "":
        return None, None

    from PIL import Image

    try:
        img_bytes = file_storage.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        risk = float(img_model.predict(img_array, verbose=0)[0][0])
        gradcam_img = make_gradcam_overlay(img_model, img_array)
        return risk, gradcam_img
    except Exception as e:
        print(f"Ultrasound image prediction error: {e}")
        return None, None


@main.route('/')
def index():
    """Landing page - project intro."""
    return render_template('index.html')


@main.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Stage 1+2: Multimodal detection with explainability.
    GET  -> show the input form (clinical fields + ultrasound upload)
    POST -> run clinical model + (optional) image model, fuse results,
            generate SHAP + Grad-CAM explanations, return everything
    """
    if request.method == 'POST':
        mdl, features = load_clinical_model()

        if mdl is None:
            return jsonify({
                'error': 'Clinical model not found. Make sure pcos_clinical_model_v2.pkl is in the models/ folder.'
            }), 500

        # ---- Read form inputs (match the 12 selected clinical features) ----
        try:
            patient_data = {
                'Follicle No. (R)': float(request.form.get('follicle_r', 0)),
                'Follicle No. (L)': float(request.form.get('follicle_l', 0)),
                'Skin darkening (Y/N)': int(request.form.get('skin_darkening', 0)),
                'hair growth(Y/N)': int(request.form.get('hair_growth', 0)),
                'Weight gain(Y/N)': int(request.form.get('weight_gain', 0)),
                'Cycle(R/I)': 1 if request.form.get('cycle', 'R') == 'I' else 0,
                'Cycle length(days)': float(request.form.get('cycle_length', 28)),
                'AMH(ng/mL)': float(request.form.get('amh', 3.0)),
                'PRL(ng/mL)': float(request.form.get('prl', 15.0)),
                'FSH/LH': float(request.form.get('fsh_lh', 1.5)),
                'Fast food (Y/N)': int(request.form.get('fast_food', 0)),
                'Pimples(Y/N)': int(request.form.get('pimples', 0)),
            }
        except (ValueError, TypeError) as e:
            return jsonify({'error': f'Invalid input: {str(e)}'}), 400

        # Validate non-negative inputs
        if patient_data['Follicle No. (R)'] < 0 or patient_data['Follicle No. (L)'] < 0:
            return jsonify({'error': 'Follicle counts cannot be negative.'}), 400
        if patient_data['Cycle length(days)'] <= 0:
            return jsonify({'error': 'Cycle length must be a positive number of days.'}), 400
        if patient_data['AMH(ng/mL)'] < 0 or patient_data['PRL(ng/mL)'] < 0 or patient_data['FSH/LH'] <= 0:
            return jsonify({'error': 'Hormone levels must be positive values.'}), 400

        # ---- Clinical model prediction ----
        X = pd.DataFrame([[patient_data[f] for f in features]], columns=features)
        clinical_risk = float(mdl.predict_proba(X)[0][1])

        # ---- SHAP explanation (Stage 2) ----
        shap_chart = generate_shap_chart(mdl, X, features)

        # ---- Image model prediction + Grad-CAM (if an ultrasound image was uploaded) ----
        uploaded_file = request.files.get('ultrasound')
        image_risk, gradcam_chart = predict_image_risk(uploaded_file)

        # ---- Decision-level fusion ----
        if image_risk is not None:
            final_risk = 0.5 * clinical_risk + 0.5 * image_risk
            modality_used = "multimodal (clinical + ultrasound)"
        else:
            final_risk = clinical_risk
            modality_used = "clinical only (no image uploaded)"

        prediction = 1 if final_risk >= 0.5 else 0

        # ---- Classify phenotype (for recommendation engine) ----
        phenotype_input = dict(patient_data)
        phenotype_input['Cycle(R/I)'] = request.form.get('cycle', 'R')
        phenotype, scores = classify_phenotype(phenotype_input)
        recommendations = get_recommendations(phenotype) if prediction == 1 else None

        # Remember this phenotype so the dashboard can show it later
        if recommendations:
            session['last_phenotype'] = phenotype

        # ---- Save this prediction to the database ----
        try:
            record = PredictionHistory(
                follicle_r=patient_data['Follicle No. (R)'],
                follicle_l=patient_data['Follicle No. (L)'],
                skin_darkening=patient_data['Skin darkening (Y/N)'],
                hair_growth=patient_data['hair growth(Y/N)'],
                weight_gain=patient_data['Weight gain(Y/N)'],
                cycle=request.form.get('cycle', 'R'),
                cycle_length=patient_data['Cycle length(days)'],
                amh=patient_data['AMH(ng/mL)'],
                prl=patient_data['PRL(ng/mL)'],
                fsh_lh=patient_data['FSH/LH'],
                fast_food=patient_data['Fast food (Y/N)'],
                pimples=patient_data['Pimples(Y/N)'],
                prediction='PCOS Detected' if prediction == 1 else 'No PCOS Detected',
                risk_score=round(final_risk, 3),
                clinical_risk_score=round(clinical_risk, 3),
                image_risk_score=round(image_risk, 3) if image_risk is not None else None,
                modality_used=modality_used,
                phenotype=recommendations['label'] if recommendations else 'N/A',
            )
            db.session.add(record)
            db.session.commit()
        except Exception as db_error:
            # Don't let a database issue break the prediction response
            print(f"Warning: could not save prediction to database: {db_error}")

        result = {
            'prediction': 'PCOS Detected' if prediction == 1 else 'No PCOS Detected',
            'risk_score': round(final_risk, 3),
            'clinical_risk_score': round(clinical_risk, 3),
            'image_risk_score': round(image_risk, 3) if image_risk is not None else None,
            'modality_used': modality_used,
            'shap_chart': shap_chart,
            'gradcam_chart': gradcam_chart,
            'phenotype': recommendations['label'] if recommendations else 'N/A',
            'phenotype_description': recommendations['description'] if recommendations else 'No significant PCOS risk detected. General healthy lifestyle recommended.',
            'recommendations': recommendations
        }
        return jsonify(result)

    return render_template('predict.html')


@main.route('/history')
def history():
    """
    View all past predictions stored in the database.
    """
    records = PredictionHistory.query.order_by(PredictionHistory.timestamp.desc()).all()
    return render_template('history.html', records=records)


@main.route('/dashboard')
def dashboard():
    """
    Stage 3: Personalized recommendation + nudge dashboard.
    Uses the phenotype from the user's most recent /predict call (if any),
    falling back to a default so the page always has something to show.
    """
    phenotype = session.get('last_phenotype', 'insulin_resistant')
    recommendations = get_recommendations(phenotype)
    return render_template('dashboard.html', rec=recommendations)
