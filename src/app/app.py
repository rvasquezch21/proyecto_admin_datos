"""
Aplicación Streamlit — Predicción de Síndrome Metabólico.

Carga el modelo entrenado (model.pkl) y permite al usuario ingresar
el perfil clínico de un paciente para obtener la predicción + probabilidad.

Ejecutar desde la raíz del proyecto:
    streamlit run src/app/app.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ──────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "ml", "model.pkl"
)

st.set_page_config(
    page_title="Predicción Síndrome Metabólico",
    page_icon="🩺",
    layout="wide",
)


# ──────────────────────────────────────────
# CARGA DEL MODELO (cacheada)
# ──────────────────────────────────────────

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


bundle = load_model()


# ──────────────────────────────────────────
# HELPERS DE FEATURES DERIVADAS
# (replicar la lógica del ETL para features que se calculan en el pipeline)
# ──────────────────────────────────────────

def bmi_category(bmi: float) -> str:
    if bmi < 18.5: return "Underweight"
    if bmi < 25.0: return "Normal"
    if bmi < 30.0: return "Overweight"
    return "Obese"

def bp_category(sys_bp: float, dia_bp: float) -> str:
    if sys_bp < 120 and dia_bp < 80: return "Normal"
    if sys_bp < 130 and dia_bp < 80: return "Elevated"
    if sys_bp < 140 or dia_bp < 90:  return "Stage1_HTN"
    return "Stage2_HTN"

def age_group(age: int) -> str:
    if age < 40: return "Young"
    if age < 65: return "Middle"
    if age < 80: return "Senior"
    return "Elderly"


# ──────────────────────────────────────────
# UI
# ──────────────────────────────────────────

st.title("🩺 Predicción de Síndrome Metabólico")
st.caption(
    "Proyecto Final — Administración de Datos · LEAD University · "
    "Modelo: RandomForest entrenado sobre `patients_curated`"
)

if bundle is None:
    st.error(
        "❌ No se encontró `model.pkl`. Ejecutá primero el entrenamiento:\n\n"
        "```bash\npython3 src/ml/train.py\n```"
    )
    st.stop()


# Sidebar — métricas del modelo
with st.sidebar:
    st.header("📊 Métricas del modelo")
    m = bundle["metrics"]
    st.metric("Accuracy",  f"{m['accuracy']*100:.1f}%")
    st.metric("ROC-AUC",   f"{m['roc_auc']:.3f}")
    st.metric("F1",        f"{m['f1']:.3f}")
    st.metric("Precision", f"{m['precision']:.3f}")
    st.metric("Recall",    f"{m['recall']:.3f}")
    st.divider()
    st.caption(f"Entrenado: {bundle['trained_at'][:19]} UTC")
    st.caption(f"N entrenamiento: {m['n_train']:,}")
    st.caption(f"N prueba: {m['n_test']:,}")
    with st.expander("Matriz de confusión"):
        cm = m["confusion_matrix"]
        cm_df = pd.DataFrame(
            cm,
            index=["Real: No", "Real: Sí"],
            columns=["Pred: No", "Pred: Sí"],
        )
        st.dataframe(cm_df)


# ──────────────────────────────────────────
# Formulario de entrada
# ──────────────────────────────────────────

st.subheader("Datos del paciente")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Demografía**")
    age = st.number_input("Edad", min_value=0, max_value=120, value=55)
    sex = st.selectbox("Sexo", ["male", "female", "other", "unknown"])
    insurance_type = st.selectbox(
        "Tipo de seguro",
        ["public", "private", "mixed", "none", "unknown"]
    )

with col2:
    st.markdown("**Mediciones clínicas**")
    bmi          = st.number_input("BMI", min_value=10.0, max_value=70.0, value=27.5, step=0.1)
    systolic_bp  = st.number_input("Presión sistólica (mmHg)",  min_value=50, max_value=300, value=130)
    diastolic_bp = st.number_input("Presión diastólica (mmHg)", min_value=30, max_value=200, value=85)
    heart_rate   = st.number_input("Frecuencia cardíaca (bpm)", min_value=20, max_value=250, value=75)
    temperature_c = st.number_input("Temperatura (°C)", min_value=32.0, max_value=43.0, value=36.8, step=0.1)

with col3:
    st.markdown("**Hábitos**")
    smoking_status = st.selectbox("Tabaquismo", ["never", "former", "current", "unknown"])
    alcohol_use    = st.selectbox("Consumo de alcohol", ["none", "moderate", "heavy", "unknown"])
    exercise_level = st.selectbox("Nivel de ejercicio", ["none", "low", "moderate", "high", "unknown"])
    charlson_index = st.number_input("Charlson index", min_value=0, max_value=20, value=2)

st.markdown("**Diagnósticos activos** (otros — los criterios del síndrome metabólico están excluidos del modelo)")

dx_cols = st.columns(4)
diagnosticos = {}
dx_options = [
    ("dx_coronary_artery_disease", "Enf. arterial coronaria"),
    ("dx_heart_failure",            "Insuficiencia cardíaca"),
    ("dx_atrial_fibrillation",      "Fibrilación auricular"),
    ("dx_chronic_kidney_disease",   "Enf. renal crónica"),
    ("dx_copd",                     "EPOC"),
    ("dx_asthma",                   "Asma"),
    ("dx_depression",               "Depresión"),
    ("dx_anxiety",                  "Ansiedad"),
    ("dx_hypothyroidism",           "Hipotiroidismo"),
    ("dx_osteoarthritis",           "Osteoartritis"),
    ("dx_type1_diabetes",           "Diabetes tipo 1"),
]
for i, (key, label) in enumerate(dx_options):
    with dx_cols[i % 4]:
        diagnosticos[key] = int(st.checkbox(label, value=False))


# ──────────────────────────────────────────
# Predicción
# ──────────────────────────────────────────

st.divider()

if st.button("🔮 Predecir", type="primary", use_container_width=True):

    # Calcular features derivadas igual que en el ETL
    pulse_pressure = systolic_bp - diastolic_bp
    map_value      = round(diastolic_bp + pulse_pressure / 3, 1)
    multimorbidity_count = sum(diagnosticos.values())  # solo los dx que entran al modelo
    cardiac_risk_flag = int(any([
        diagnosticos.get("dx_coronary_artery_disease", 0),
        diagnosticos.get("dx_heart_failure", 0),
        diagnosticos.get("dx_atrial_fibrillation", 0),
    ]))

    record = {
        "age": age,
        "bmi": bmi,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "heart_rate": heart_rate,
        "temperature_c": temperature_c,
        "charlson_index": charlson_index,
        "multimorbidity_count": multimorbidity_count,
        "pulse_pressure": pulse_pressure,
        "map": map_value,
        "sex": sex,
        "smoking_status": smoking_status,
        "alcohol_use": alcohol_use,
        "exercise_level": exercise_level,
        "insurance_type": insurance_type,
        "bmi_category": bmi_category(bmi),
        "bp_category":  bp_category(systolic_bp, diastolic_bp),
        "age_group":    age_group(age),
        "cardiac_risk_flag": cardiac_risk_flag,
        **diagnosticos,
    }

    # Reordenar según el modelo
    X_input = pd.DataFrame([record])[bundle["feature_cols"]]

    model = bundle["model"]
    proba = float(model.predict_proba(X_input)[0, 1])
    pred  = int(proba >= 0.5)

    st.subheader("Resultado")
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        if pred == 1:
            st.error(f"### ⚠️ Síndrome metabólico: SÍ\n**Probabilidad: {proba*100:.1f}%**")
        else:
            st.success(f"### ✅ Síndrome metabólico: NO\n**Probabilidad: {proba*100:.1f}%**")

    with res_col2:
        st.markdown("**Interpretación**")
        if proba < 0.3:
            st.write("Riesgo bajo. Perfil clínico sin señales fuertes de síndrome metabólico.")
        elif proba < 0.6:
            st.write("Riesgo intermedio. Conviene seguimiento de factores modificables (peso, presión, hábitos).")
        else:
            st.write("Riesgo alto. Recomendable evaluación clínica formal de los criterios del síndrome metabólico.")

        st.progress(proba)

    with st.expander("Ver features enviadas al modelo"):
        st.json(record)
