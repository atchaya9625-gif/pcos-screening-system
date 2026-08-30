"""
Stage 3: Personalized Recommendation Engine
=============================================
Idhu module enna pannum:
1. Model prediction + patient inputs base pannitu, PCOS phenotype classify pannum
   (Insulin-Resistant / Inflammatory-Androgen / Cycle-Irregularity)
2. Andha phenotype ku match aagura time-slotted (morning/afternoon/evening/night)
   food + yoga/exercise recommendations return pannum
"""

# ============================================================
# PHENOTYPE CLASSIFICATION LOGIC
# ============================================================

def classify_phenotype(patient_data):
    """
    patient_data: dict with keys matching the 12 selected clinical features:
        'Weight gain(Y/N)', 'Fast food (Y/N)', 'Skin darkening (Y/N)',
        'hair growth(Y/N)', 'Pimples(Y/N)', 'Cycle(R/I)', 'Cycle length(days)',
        'Follicle No. (L)', 'Follicle No. (R)', 'AMH(ng/mL)', 'PRL(ng/mL)', 'FSH/LH'

    Returns: one of "insulin_resistant", "inflammatory_androgen", "cycle_irregularity"
    (simple rule-based scoring - counts symptoms matching each phenotype, picks the highest)
    """
    insulin_score = 0
    androgen_score = 0
    cycle_score = 0

    # --- Insulin-Resistant signals ---
    if patient_data.get("Weight gain(Y/N)", 0) == 1:
        insulin_score += 1
    if patient_data.get("Fast food (Y/N)", 0) == 1:
        insulin_score += 1

    # --- Inflammatory / Androgen signals ---
    if patient_data.get("Skin darkening (Y/N)", 0) == 1:
        androgen_score += 1
    if patient_data.get("hair growth(Y/N)", 0) == 1:
        androgen_score += 1
    if patient_data.get("Pimples(Y/N)", 0) == 1:
        androgen_score += 1

    # --- Cycle-Irregularity signals ---
    if patient_data.get("Cycle(R/I)") in ["I", 1]:  # I = Irregular
        cycle_score += 1
    cycle_length = patient_data.get("Cycle length(days)", 28)
    if cycle_length < 21 or cycle_length > 35:
        cycle_score += 1

    # --- Hormonal tiebreakers (used if scores are close) ---
    fsh_lh = patient_data.get("FSH/LH", 1.0)
    if fsh_lh < 1.0:  # LH >> FSH is a classic PCOS androgen marker
        androgen_score += 0.5

    amh = patient_data.get("AMH(ng/mL)", 3.0)
    if amh > 4.5:  # high AMH often correlates with follicle count / insulin resistance
        insulin_score += 0.5

    scores = {
        "insulin_resistant": insulin_score,
        "inflammatory_androgen": androgen_score,
        "cycle_irregularity": cycle_score,
    }

    # Pick the highest scoring phenotype (default to insulin_resistant if all zero)
    phenotype = max(scores, key=scores.get)
    if all(v == 0 for v in scores.values()):
        phenotype = "insulin_resistant"

    return phenotype, scores


# ============================================================
# RECOMMENDATION CONTENT (time-slotted, per phenotype)
# ============================================================

RECOMMENDATIONS = {
    "insulin_resistant": {
        "label": "Insulin-Resistant / Weight-Focused",
        "description": "High BMI, weight gain, and insulin-related markers detected. Recommendations focus on metabolism and blood sugar regulation.",
        "morning": {
            "food": "Protein-rich breakfast (eggs, paneer, or moong dal chilla), avoid sugary tea/coffee",
            "exercise": "Surya Namaskar (Sun Salutation, 5-10 rounds) + 30 min brisk walk"
        },
        "afternoon": {
            "food": "Low-glycemic meal - brown rice or millets with vegetables and dal, avoid white rice/sugary drinks",
            "exercise": "Malasana (Garland Pose) - hold 1-2 min, helps insulin sensitivity"
        },
        "evening": {
            "food": "Light snack - roasted chana, nuts, or sprouts; avoid fried snacks",
            "exercise": "Chakki Chalanasana (Mill Churning Pose) + light strength training, 3x/week"
        },
        "night": {
            "food": "Early dinner before 8 PM, avoid late-night carbs and sugary desserts",
            "exercise": "Dhanurasana (Bow Pose) - gentle, 3-4 rounds, then wind down"
        }
    },
    "inflammatory_androgen": {
        "label": "Inflammatory / Androgen-Related",
        "description": "Acne, hair growth, or skin darkening detected - markers linked to higher androgen levels. Recommendations focus on hormonal balance and inflammation.",
        "morning": {
            "food": "Anti-inflammatory foods - turmeric milk, berries, flaxseeds",
            "exercise": "Baddha Konasana (Butterfly Pose) - 3-5 min, stimulates ovarian function"
        },
        "afternoon": {
            "food": "Omega-3 rich meal - fish or walnuts with salad, avoid processed/fried food",
            "exercise": "Bhujangasana (Cobra Pose) - 5 rounds, supports adrenal balance"
        },
        "evening": {
            "food": "Spearmint tea (shown to help with excess hair growth/acne)",
            "exercise": "Ustrasana (Camel Pose) - supports thyroid and adrenal function"
        },
        "night": {
            "food": "Turmeric milk before bed, avoid dairy-heavy or oily dinners",
            "exercise": "Shavasana + Pranayama (deep breathing, 5-10 min) for stress/cortisol control"
        }
    },
    "cycle_irregularity": {
        "label": "Cycle-Irregularity Focused",
        "description": "Irregular or unusually long/short menstrual cycles detected. Recommendations focus on hormonal and cycle regulation.",
        "morning": {
            "food": "Seed cycling - pumpkin + flax seeds (follicular phase), iron-rich foods if flow is heavy",
            "exercise": "Setu Bandhasana (Bridge Pose) - improves pelvic blood circulation"
        },
        "afternoon": {
            "food": "Balanced meal with healthy fats - avocado, olive oil, nuts for hormone production",
            "exercise": "Baddha Konasana (Butterfly Pose) - direct benefit for reproductive organs"
        },
        "evening": {
            "food": "Herbal tea (chamomile or ginger), avoid high-intensity dieting",
            "exercise": "Moderate walk or swim, 20-30 min - avoid high-intensity if cycle very irregular"
        },
        "night": {
            "food": "Magnesium-rich foods - banana, dark chocolate (small portion) for cycle regulation",
            "exercise": "Anulom Vilom Pranayama (alternate nostril breathing) - stress reduction, cycle support"
        }
    }
}


def get_recommendations(phenotype):
    """Returns the full time-slotted recommendation dict for a given phenotype."""
    return RECOMMENDATIONS.get(phenotype, RECOMMENDATIONS["insulin_resistant"])
