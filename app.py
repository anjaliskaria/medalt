import streamlit as st
import pickle
import re
import numpy as np
from scipy.sparse import hstack
import nltk


# ============================================================
# 1. Page Config
# ============================================================
st.set_page_config(
    page_title="MedAlt — Drug Predictor",
    page_icon="💊",
    layout="centered"
)
 
# ============================================================
# 2. Download stopwords (cached so it only runs once)
# ============================================================
@st.cache_resource
def load_stopwords():
    nltk.download("stopwords")
    from nltk.corpus import stopwords as nltk_stopwords
    return set(nltk_stopwords.words("english"))
 
stop_words = load_stopwords()
 
# ============================================================
# 3. Load All Saved Models (cached so it only loads once)
# ============================================================
@st.cache_resource
def load_models():
    lr_model = pickle.load(open("layer1_model.pkl", "rb"))
    xgb_model = pickle.load(open("layer2_model.pkl", "rb"))
    tfidf = pickle.load(open("tfidf.pkl", "rb"))
    le = pickle.load(open("label_encoder.pkl", "rb"))
    mlb = pickle.load(open("mlb.pkl", "rb"))
    return lr_model, xgb_model, tfidf, le, mlb
 
try:
    lr_model, xgb_model, tfidf, le, mlb = load_models()
    models_loaded = True
except FileNotFoundError as e:
    models_loaded = False
    missing_file = str(e)
 
# ============================================================
# 4. Text Cleaning Function (same as Module 4)
# ============================================================
def clean_text(text):
    return " ".join(
        w for w in re.sub(r"[^a-z\s]", "", text.lower()).split()
        if w not in stop_words
    )
 
# ============================================================
# 5. Prediction Pipeline Function (from Module 8)
# ============================================================
def predict_drug(uses, chemical_class, action_class, habit):
    # Combine text features
    combined = uses + " " + chemical_class + " " + action_class
    combined_cleaned = clean_text(combined)
 
    # TF-IDF transform
    X_input = tfidf.transform([combined_cleaned])
 
    # Add Habit Forming feature
    X_input_combined = hstack([X_input, [[habit]]])
 
    # Layer 1 — Therapeutic Class
    y_pred_class = lr_model.predict(X_input)
    therapeutic_class = le.inverse_transform(y_pred_class)[0]
 
    # Layer 2 — Side Effects
    # y_pred_se = xgb_model.predict(X_input_combined)
    # side_effects = mlb.inverse_transform(y_pred_se)[0]
    y_pred_proba = xgb_model.predict_proba(X_input_combined)
    y_pred_se = (y_pred_proba >= 0.3).astype(int)
    side_effects = mlb.inverse_transform(y_pred_se)[0]
 
    return therapeutic_class, side_effects
 
# ============================================================
# 6. Streamlit UI
# ============================================================
 
st.title("💊 MedAlt")
st.markdown("##### Therapeutic Class Predictor & Side Effect Classifier")
st.divider()
 
if not models_loaded:
    st.error(
        f"⚠️ Model files not found. Make sure these 5 files are in the "
        f"same folder as app.py:\n\n"
        f"- layer1_model.pkl\n- layer2_model.pkl\n- tfidf.pkl\n"
        f"- label_encoder.pkl\n- mlb.pkl"
    )
    st.stop()
 
st.markdown("Enter the drug details below to predict its **therapeutic class** "
            "and **likely side effects**.")
 
with st.form("drug_form"):
    drug_name = st.text_input(
        "Drug Name (optional, for display only)",
        placeholder="e.g. Amoxicillin"
    )
 
    uses = st.text_area(
        "Uses",
        placeholder="e.g. treats bacterial infections pneumonia ear infection"
    )
 
    chemical_class = st.text_input(
        "Chemical Class",
        placeholder="e.g. beta lactam penicillin"
    )
 
    action_class = st.text_input(
        "Action Class",
        placeholder="e.g. cell wall inhibitor bactericidal"
    )
 
    habit_forming = st.selectbox(
        "Is this drug Habit Forming?",
        ["No", "Yes"]
    )
 
    submitted = st.form_submit_button("🔍 Predict")
 
# ============================================================
# 7. Handle Prediction
# ============================================================
if submitted:
    if not uses.strip():
        st.warning("Please enter the **Uses** field — it's required for prediction.")
    else:
        habit = 1 if habit_forming == "Yes" else 0
 
        with st.spinner("Predicting..."):
            therapeutic_class, side_effects = predict_drug(
                uses, chemical_class, action_class, habit
            )
 
        st.divider()
        if drug_name.strip():
            st.subheader(f"📋 Results for {drug_name}")
        else:
            st.subheader("📋 Prediction Results")
 
        st.markdown(f"**Therapeutic Class:**")
        st.success(therapeutic_class)
 
        st.markdown("**Likely Side Effects:**")
        if len(side_effects) == 0:
            st.info("No significant side effects predicted with high confidence.")
        else:
            for se in side_effects:
                st.markdown(f"- {se}")
 
# # ============================================================
# # 8. Footer
# # ============================================================
# st.divider()
# st.caption(
#     "MedAlt — Layer 1: Logistic Regression (96% accuracy) | "
#     "Layer 2: XGBoost OneVsRest (Jaccard 0.76)"
# )