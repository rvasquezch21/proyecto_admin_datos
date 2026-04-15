"""
Entrenamiento del modelo de IA — Clasificación de síndrome metabólico.

Lee patients_curated desde MongoDB, entrena un RandomForest, evalúa
con métricas estándar y serializa el modelo + metadata a model.pkl.

Excluye explícitamente las columnas que componen la fórmula del target
para evitar data leakage (ver README sección "Modelo IA").
"""

import sys
import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Importar conexión Mongo del proyecto
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "db"))
from connection import get_database


# ──────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────

SOURCE_COLLECTION = "patients_curated"
MODEL_DIR         = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH        = os.path.join(MODEL_DIR, "model.pkl")
METRICS_PATH      = os.path.join(MODEL_DIR, "metrics.json")

TARGET = "metabolic_syndrome"

# Columnas EXCLUIDAS por leakage (componen la fórmula del target en el ETL)
LEAKAGE_COLS = [
    "metabolic_syndrome",        # el target
    "metabolic_criteria_count",  # conteo determinístico
    "dx_obesity",                # criterio 1
    "dx_hypertension",           # criterio 2
    "dx_type2_diabetes",         # criterio 3
    "dx_hyperlipidemia",         # criterio 4
    # bmi se mantiene como numérico continuo (aporta info más allá del corte ≥30)
]

# Columnas que tampoco entran al modelo (identificadores / metadata)
META_COLS = [
    "_id", "patient_id", "etl_timestamp",
    "data_quality_flag", "data_quality_issues",
    # Derivadas que también podrían filtrar info del target indirectamente:
    "risk_score", "complexity_tier",  # se basan en multimorbidity_count que sí dejamos
]

NUMERIC_FEATURES = [
    "age", "bmi", "systolic_bp", "diastolic_bp", "heart_rate",
    "temperature_c", "charlson_index", "multimorbidity_count",
    "pulse_pressure", "map",
]

CATEGORICAL_FEATURES = [
    "sex", "smoking_status", "alcohol_use", "exercise_level",
    "insurance_type", "bmi_category", "bp_category", "age_group",
]

BINARY_DX_FEATURES = [
    "dx_coronary_artery_disease", "dx_heart_failure", "dx_atrial_fibrillation",
    "dx_chronic_kidney_disease", "dx_copd", "dx_asthma",
    "dx_depression", "dx_anxiety", "dx_hypothyroidism",
    "dx_osteoarthritis", "dx_type1_diabetes",
    "cardiac_risk_flag",
]


# ──────────────────────────────────────────
# CARGA DE DATOS
# ──────────────────────────────────────────

def load_dataset() -> pd.DataFrame:
    print(f"\n[CARGA] Leyendo {SOURCE_COLLECTION} desde MongoDB…")
    db = get_database()
    cursor = db[SOURCE_COLLECTION].find({}, {"_id": 0})
    df = pd.DataFrame(list(cursor))
    print(f"   ✅ {len(df):,} registros — {len(df.columns)} columnas")
    return df


def prepare_features(df: pd.DataFrame):
    print("\n[PREP] Separando features y target…")

    # Filtrar filas con flag de calidad si quieren modelo "limpio"
    if "data_quality_flag" in df.columns:
        n_before = len(df)
        df = df[df["data_quality_flag"] != True].copy()
        print(f"   {n_before - len(df):,} filas filtradas por data_quality_flag")

    y = df[TARGET].astype(int)

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_DX_FEATURES
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].copy()

    print(f"   ✅ X: {X.shape}, y: {y.shape}")
    print(f"   Distribución target: positivos={int(y.sum()):,} ({y.mean()*100:.1f}%)")
    return X, y, feature_cols


# ──────────────────────────────────────────
# PIPELINE DE MODELO
# ──────────────────────────────────────────

def build_pipeline(numeric_cols, categorical_cols, binary_cols) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("bin", "passthrough", binary_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    return Pipeline([("prep", preprocessor), ("clf", model)])


# ──────────────────────────────────────────
# ENTRENAMIENTO Y EVALUACIÓN
# ──────────────────────────────────────────

def train_and_evaluate(X, y, feature_cols):
    print("\n[ENTRENAMIENTO] Split 80/20 estratificado…")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    num_in_X = [c for c in NUMERIC_FEATURES if c in X.columns]
    cat_in_X = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    bin_in_X = [c for c in BINARY_DX_FEATURES if c in X.columns]

    pipe = build_pipeline(num_in_X, cat_in_X, bin_in_X)

    print("   Entrenando RandomForest…")
    pipe.fit(X_train, y_train)

    print("\n[EVALUACIÓN]")
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_train": int(len(X_train)),
        "n_test":  int(len(X_test)),
        "positive_rate_train": round(float(y_train.mean()), 4),
        "positive_rate_test":  round(float(y_test.mean()), 4),
    }

    print(f"   Accuracy : {metrics['accuracy']}")
    print(f"   Precision: {metrics['precision']}")
    print(f"   Recall   : {metrics['recall']}")
    print(f"   F1       : {metrics['f1']}")
    print(f"   ROC-AUC  : {metrics['roc_auc']}")
    print(f"\n   Matriz de confusión [[TN, FP], [FN, TP]]:")
    print(f"   {metrics['confusion_matrix']}")
    print("\n   Reporte detallado:")
    print(classification_report(y_test, y_pred, target_names=["No-MetSyn", "MetSyn"]))

    # Validación cruzada (rápida, 3-fold) sobre el set completo
    print("\n[CV] 3-fold cross-val ROC-AUC sobre todo el dataset…")
    cv_scores = cross_val_score(pipe, X, y, cv=3, scoring="roc_auc", n_jobs=-1)
    metrics["cv_roc_auc_mean"] = round(float(cv_scores.mean()), 4)
    metrics["cv_roc_auc_std"]  = round(float(cv_scores.std()), 4)
    print(f"   ROC-AUC CV: {metrics['cv_roc_auc_mean']} ± {metrics['cv_roc_auc_std']}")

    return pipe, metrics


# ──────────────────────────────────────────
# PERSISTENCIA
# ──────────────────────────────────────────

def save_artifacts(pipe, metrics, feature_cols):
    bundle = {
        "model": pipe,
        "feature_cols": feature_cols,
        "numeric_features":     [c for c in NUMERIC_FEATURES     if c in feature_cols],
        "categorical_features": [c for c in CATEGORICAL_FEATURES if c in feature_cols],
        "binary_features":      [c for c in BINARY_DX_FEATURES   if c in feature_cols],
        "target": TARGET,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\n   💾 Modelo guardado en: {MODEL_PATH}")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"   💾 Métricas guardadas en: {METRICS_PATH}")


# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────

def main():
    print("=" * 55)
    print(" ENTRENAMIENTO MODELO IA — Síndrome Metabólico")
    print(f" Inicio: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 55)

    df = load_dataset()
    X, y, feature_cols = prepare_features(df)
    pipe, metrics = train_and_evaluate(X, y, feature_cols)
    save_artifacts(pipe, metrics, feature_cols)

    print("\n" + "=" * 55)
    print("  ENTRENAMIENTO COMPLETADO")
    print(f"   Fin: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 55)


if __name__ == "__main__":
    main()
