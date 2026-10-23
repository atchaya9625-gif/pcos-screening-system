import os
import io
import gc
import base64
import joblib
import traceback
import numpy as np
import pandas as pd
import matplotlib

# Optimize TensorFlow and Matplotlib for headless cloud containers
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Blueprint, render_template, request, jsonify, session

from app.recommendations import classify_phenotype, get_recommendations
from app.models import db, PredictionHistory

main = Blueprint('main', __name__)

# ============================================================
# Paths & Model Configuration
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLINICAL_MODEL_PATH = os.path.join(BASE_DIR, "models", "pcos_clinical_model_v2.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "selected_features_v2.pkl")
KERAS_MODEL_PATH = os.path.join(BASE_DIR, "models", "pcos_image_model.keras")
TFLITE_MODEL_PATH = os.path.join(BASE_DIR, "models", "pcos_image_model.tflite")

IMG_SIZE = 160

# Lazy global model singletons
clinical_model = None
selected_features = None
image_keras_model = None
image_tflite_interpreter = None
shap_explainer = None


# ============================================================
# Model Loaders
# ============================================================
def load_clinical_model():
    """Load the trained XGBoost clinical model and 12 MI-selected features."""
    global clinical_model, selected_features
    if clinical_model is None and os.path.exists(CLINICAL_MODEL_PATH):
        clinical_model = joblib.load(CLINICAL_MODEL_PATH)
        selected_features = joblib.load(FEATURES_PATH)
    return clinical_model, selected_features


def load_image_model():
    """
    Load the ultrasound deep learning model.
    Tries TensorFlow/Keras model first (enables Grad-CAM explainability).
    Falls back to LiteRT / TFLite interpreter if TensorFlow is unavailable or encounters memory limits.
    """
    global image_keras_model, image_tflite_interpreter

    # 1. Try Keras model for full inference + Grad-CAM
    if image_keras_model is None and os.path.exists(KERAS_MODEL_PATH):
        try:
            import tensorflow as tf
            tf.get_logger().setLevel('ERROR')
            image_keras_model = tf.keras.models.load_model(KERAS_MODEL_PATH)
            return image_keras_model, "keras"
        except Exception as e:
            print(f"[Image Model] TensorFlow/Keras load error: {e}. Attempting TFLite fallback...")

    if image_keras_model is not None:
        return image_keras_model, "keras"

    # 2. Fallback to lightweight TFLite model if available
    if image_tflite_interpreter is None and os.path.exists(TFLITE_MODEL_PATH):
        try:
            from ai_edge_litert.interpreter import Interpreter
            image_tflite_interpreter = Interpreter(model_path=TFLITE_MODEL_PATH)
            image_tflite_interpreter.allocate_tensors()
            return image_tflite_interpreter, "tflite"
        except Exception as e:
            print(f"[Image Model] TFLite load error: {e}")

    if image_tflite_interpreter is not None:
        return image_tflite_interpreter, "tflite"

    return None, None


def get_shap_explainer(mdl):
    """Retrieve or initialize the TreeExplainer for the XGBoost model."""
    global shap_explainer
    if shap_explainer is None:
        import shap
        shap_explainer = shap.TreeExplainer(mdl)
    return shap_explainer


def fig_to_base64(fig):
    """Convert a matplotlib figure to a base64 PNG data URI string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


# ============================================================
# Explainable AI (XAI): SHAP & Grad-CAM
# ============================================================
def generate_shap_chart(mdl, X, features):
    """Generate a horizontal bar chart showing individual feature contributions via SHAP."""
    try:
        explainer = get_shap_explainer(mdl)
        shap_values = explainer.shap_values(X)

        values = shap_values[0] if shap_values.ndim == 2 else shap_values
        contributions = pd.Series(values, index=features).sort_values(key=abs)

        # Pink/purple theme matching UI: #D6336C (increases PCOS risk), #6C5CE7 (protective/lowers risk)
        colors = ["#D6336C" if v > 0 else "#6C5CE7" for v in contributions.values]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(contributions.index, contributions.values, color=colors)
        ax.set_xlabel("Impact on PCOS risk (SHAP value)", fontsize=10, color="#3d2645")
        ax.set_title("Why this prediction? (Clinical features)", fontsize=11, fontweight="bold", color="#a3336b")
        ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        b64 = fig_to_base64(fig)
        plt.close('all')
        return b64
    except Exception as e:
        print(f"[XAI] SHAP generation error: {e}")
        return None


def make_gradcam_overlay(img_model, img_array_norm):
    """
    Generate Grad-CAM heatmap overlay for the ultrasound scan using EfficientNetB0's top_conv layer.
    """
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
        conv_output_val = conv_output[0]
        heatmap = conv_output_val @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0

        original = img_array_norm[0]
        overlay = 0.6 * original + 0.4 * heatmap_colored
        overlay = np.clip(overlay, 0.0, 1.0)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(original)
        axes[0].set_title("Ultrasound Scan", fontsize=11, fontweight="bold", color="#3d2645")
        axes[0].axis("off")

        axes[1].imshow(overlay)
        axes[1].set_title("Grad-CAM: Model Focus Area", fontsize=11, fontweight="bold", color="#a3336b")
        axes[1].axis("off")

        plt.tight_layout()
        b64 = fig_to_base64(fig)
        plt.close('all')
        gc.collect()
        return b64
    except Exception as e:
        print(f"[XAI] Grad-CAM error: {e}")
        return None


# ============================================================
# Ultrasound Image Inference Pipeline
# ============================================================
def predict_image_risk(file_storage):
    """
    Takes an uploaded ultrasound scan file, preprocesses it, and runs inference.
    Returns (risk_score, gradcam_b64_or_None).
    """
    if file_storage is None or not getattr(file_storage, 'filename', ''):
        return None, None

    model_obj, model_type = load_image_model()
    if model_obj is None:
        print("[Image Model] No ultrasound model available.")
        return None, None

    from PIL import Image

    try:
        img_bytes = file_storage.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if model_type == "keras":
            raw_pred = model_obj.predict(img_array, verbose=0)
            risk = float(raw_pred[0][0])
            gradcam_img = make_gradcam_overlay(model_obj, img_array)
            return risk, gradcam_img
        elif model_type == "tflite":
            input_details = model_obj.get_input_details()[0]
            output_details = model_obj.get_output_details()[0]
            model_obj.set_tensor(input_details["index"], img_array)
            model_obj.invoke()
            output = model_obj.get_tensor(output_details["index"])
            risk = float(output[0][0])
            return risk, None
    except Exception as e:
        print(f"[Image Model] Inference error: {e}")
        return None, None


# ============================================================
# Routes
# ============================================================
@main.route('/')
def index():
    """Landing page - project highlights, methodology overview, navigation."""
    return render_template('index.html')


@main.route('/health')
def health():
    """Diagnostic endpoint to inspect runtime health, models, and dependencies."""
    status = {
        'status': 'ok',
        'clinical_model': os.path.exists(CLINICAL_MODEL_PATH),
        'features_file': os.path.exists(FEATURES_PATH),
        'keras_model': os.path.exists(KERAS_MODEL_PATH),
        'tflite_model': os.path.exists(TFLITE_MODEL_PATH),
    }
    try:
        import tensorflow as tf
        status['tensorflow_version'] = tf.__version__
    except Exception as e:
        status['tensorflow_error'] = str(e)

    try:
        from ai_edge_litert.interpreter import Interpreter
        status['litert_available'] = True
    except Exception as e:
        status['litert_error'] = str(e)

    return jsonify(status)


@main.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Stage 1 + 2: Multimodal detection & explainability.
    GET: Renders clinical input form with preset loader and optional image upload.
    POST: Runs XGBoost + CNN fusion, produces SHAP & Grad-CAM, classifies phenotype,
          stores record in SQLite, and returns structured JSON.
    """
    if request.method == 'POST':
        try:
            mdl, features = load_clinical_model()
            if mdl is None:
                return jsonify({
                    'error': 'Clinical model not loaded. Ensure pcos_clinical_model_v2.pkl is present in models/.'
                }), 500

            # Parse form inputs
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
                return jsonify({'error': f'Invalid input format: {str(e)}'}), 400

            # Input bounds validation
            if patient_data['Follicle No. (R)'] < 0 or patient_data['Follicle No. (L)'] < 0:
                return jsonify({'error': 'Follicle counts cannot be negative.'}), 400
            if patient_data['Follicle No. (R)'] > 80 or patient_data['Follicle No. (L)'] > 80:
                return jsonify({'error': 'Follicle counts must be within realistic clinical range (0 to 80).'}), 400

            if patient_data['Cycle length(days)'] <= 0:
                return jsonify({'error': 'Cycle length must be a positive number of days.'}), 400
            if patient_data['Cycle length(days)'] < 15 or patient_data['Cycle length(days)'] > 180:
                return jsonify({'error': 'Cycle length must be within realistic clinical range (15 to 180 days).'}), 400

            if patient_data['AMH(ng/mL)'] < 0 or patient_data['PRL(ng/mL)'] < 0 or patient_data['FSH/LH'] <= 0:
                return jsonify({'error': 'Hormone levels must be positive values.'}), 400
            if patient_data['AMH(ng/mL)'] > 100:
                return jsonify({'error': 'AMH level exceeds maximum realistic clinical range (0 to 100 ng/mL).'}), 400
            if patient_data['PRL(ng/mL)'] > 300:
                return jsonify({'error': 'Prolactin (PRL) level exceeds maximum realistic clinical range (0 to 300 ng/mL).'}), 400
            if patient_data['FSH/LH'] > 30:
                return jsonify({'error': 'FSH/LH ratio exceeds maximum realistic clinical range (0.01 to 30.0).'}), 400

            # 1. Clinical Model Prediction
            X = pd.DataFrame([[patient_data[f] for f in features]], columns=features)
            clinical_risk = float(mdl.predict_proba(X)[0][1])

            # 2. Clinical Explainability (SHAP)
            shap_chart = generate_shap_chart(mdl, X, features)

            # 3. Ultrasound Model Prediction + Grad-CAM
            uploaded_file = request.files.get('ultrasound')
            image_risk, gradcam_chart = predict_image_risk(uploaded_file)

            # 4. Multimodal Decision-Level Fusion
            if image_risk is not None:
                final_risk = 0.5 * clinical_risk + 0.5 * image_risk
                modality_used = "multimodal (clinical + ultrasound)"
            else:
                final_risk = clinical_risk
                modality_used = "clinical only (no image uploaded)"

            prediction = 1 if final_risk >= 0.5 else 0

            # 5. Phenotype Classification & Continuous Lifestyle Plan
            phenotype_input = dict(patient_data)
            phenotype_input['Cycle(R/I)'] = request.form.get('cycle', 'R')
            phenotype, scores = classify_phenotype(phenotype_input)
            recommendations = get_recommendations(phenotype) if prediction == 1 else None

            # Cache phenotype in session for dashboard access
            if recommendations:
                session['last_phenotype'] = phenotype

            # 6. Database Persistence (India Standard Time)
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
                print(f"[Database] Warning: could not save prediction: {db_error}")

            # 7. Construct Structured JSON Response
            result = {
                'prediction': 'PCOS Detected' if prediction == 1 else 'No PCOS Detected',
                'risk_score': round(final_risk, 3),
                'clinical_risk_score': round(clinical_risk, 3),
                'image_risk_score': round(image_risk, 3) if image_risk is not None else None,
                'modality_used': modality_used,
                'shap_chart': shap_chart,
                'gradcam_chart': gradcam_chart,
                'phenotype': recommendations['label'] if recommendations else 'N/A',
                'phenotype_description': (
                    recommendations['description']
                    if recommendations
                    else 'No significant PCOS risk detected. General healthy lifestyle and routine health maintenance recommended.'
                ),
                'recommendations': recommendations
            }
            return jsonify(result)

        except Exception as e:
            trace = traceback.format_exc()
            print(f"[Server Error in /predict]:\n{trace}")
            return jsonify({'error': f'Internal analysis error: {str(e)}', 'trace': trace}), 500

    return render_template('predict.html')


@main.route('/history')
def history():
    """View all past screening predictions stored in the database."""
    records = PredictionHistory.query.order_by(PredictionHistory.timestamp.desc()).all()
    return render_template('history.html', records=records)


@main.route('/dashboard')
def dashboard():
    """
    Stage 3: Personalized recommendation + nudge dashboard.
    Uses the phenotype from the user's most recent /predict call,
    falling back to a default so the page always displays actionable guidance.
    """
    phenotype = session.get('last_phenotype', 'insulin_resistant')
    recommendations = get_recommendations(phenotype)
    return render_template('dashboard.html', rec=recommendations)
