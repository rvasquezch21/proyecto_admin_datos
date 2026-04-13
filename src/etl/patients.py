import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pymongo import UpdateOne

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "db"))

from connection import get_database

#  CONFIGURACIÓN

SOURCE_COLLECTION = "patients"
TARGET_COLLECTION = "patients_curated"
CHUNK_SIZE        = 5000

DX_COLUMNS = [
    "dx_hypertension", "dx_type2_diabetes", "dx_hyperlipidemia",
    "dx_obesity", "dx_coronary_artery_disease", "dx_heart_failure",
    "dx_atrial_fibrillation", "dx_chronic_kidney_disease", "dx_copd",
    "dx_asthma", "dx_depression", "dx_anxiety", "dx_hypothyroidism",
    "dx_osteoarthritis", "dx_type1_diabetes",
]

NUMERIC_COLS     = ["age", "bmi", "systolic_bp", "diastolic_bp", "heart_rate", "temperature_f", "charlson_index"]
CATEGORICAL_COLS = ["sex", "smoking_status", "alcohol_use", "exercise_level", "insurance_type"]

CLINICAL_RANGES = {
    "age":           (0,   120),
    "bmi":           (10,   70),
    "systolic_bp":   (50,  300),
    "diastolic_bp":  (30,  200),
    "heart_rate":    (20,  250),
    "temperature_f": (90,  110),
}


#  EXTRACCIÓN

def extract(db) -> pd.DataFrame:
    print("\n[EXTRACCIÓN] Leyendo colección:", SOURCE_COLLECTION)

    collection = db[SOURCE_COLLECTION]
    total = collection.count_documents({})
    print(f"   {total:,} documentos en la colección, leyendo por lotes...")

    records = []
    batch_size = 5000
    cursor = collection.find({}, {"_id": 0}).batch_size(batch_size)
    
    for doc in cursor:
        records.append(doc)
        if len(records) % batch_size == 0:
            print(f"   📥 {len(records):,}/{total:,} documentos leídos...")

    df = pd.DataFrame(records)
    print(f"   ✅ {len(df):,} documentos extraídos — {len(df.columns)} columnas")
    return df


#  TRANSFORMACIÓN — LIMPIEZA

def clean(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[LIMPIEZA] Iniciando transformaciones de calidad…")
    df        = df.copy()
    flags_log = []

    # 1. Duplicados
    before  = len(df)
    df      = df.drop_duplicates(subset="patient_id", keep="first")
    dropped = before - len(df)
    if dropped:
        print(f"    {dropped} duplicados eliminados por patient_id")

    # 2. Casteo de tipos
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["age"] = df["age"].astype("Int64")

    for col in DX_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # 3. Imputar nulos
    for col in NUMERIC_COLS:
        if col in df.columns:
            n_null = df[col].isna().sum()
            if n_null:
                median_val = df[col].median()
                df[col]    = df[col].fillna(median_val)
                print(f"   {col}: {n_null} nulos → imputados con mediana ({median_val:.2f})")

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            n_null = df[col].isna().sum()
            if n_null:
                df[col] = df[col].fillna("unknown")
                print(f"   {col}: {n_null} nulos → imputados con 'unknown'")

    # 4. Validación clínica — diastólica < sistólica
    inconsistent_bp = df["diastolic_bp"] >= df["systolic_bp"]
    n_bp = inconsistent_bp.sum()
    if n_bp:
        print(f"     {n_bp} pacientes con diastólica >= sistólica (inconsistencia clínica)")

    # 5. Validar diabetes tipo 1 y tipo 2 simultáneas
    if "dx_type1_diabetes" in df.columns and "dx_type2_diabetes" in df.columns:
        both_diabetes = (df["dx_type1_diabetes"] == 1) & (df["dx_type2_diabetes"] == 1)
        n_both = both_diabetes.sum()
        if n_both:
            print(f"     {n_both} pacientes con DM1 y DM2 simultáneas (revisar)")

    # 6. Rangos clínicos → data_quality_flag
    df["data_quality_flag"]   = False
    df["data_quality_issues"] = ""

    for col, (lo, hi) in CLINICAL_RANGES.items():
        if col not in df.columns:
            continue
        numeric_col  = pd.to_numeric(df[col], errors="coerce")
        out_of_range = (numeric_col < lo) | (numeric_col > hi)
        n_bad        = out_of_range.sum()
        if n_bad:
            df.loc[out_of_range, "data_quality_flag"]    = True
            df.loc[out_of_range, "data_quality_issues"] += f"{col}_out_of_range; "
            flags_log.append(f"      {col}: {n_bad} valores fuera de [{lo}, {hi}]")

    n_flagged = df["data_quality_flag"].sum()
    if flags_log:
        print(f"   🚩 {n_flagged} filas marcadas con data_quality_flag:")
        for entry in flags_log:
            print(entry)
    else:
        print("   Todos los rangos clínicos son válidos")

    print(f"   Limpieza completa — {len(df):,} registros resultantes")
    return df


#  TRANSFORMACIÓN — FEATURE ENGINEERING

def _bmi_category(bmi: float) -> str:
    if pd.isna(bmi): return "Unknown"
    if bmi < 18.5:   return "Underweight"
    if bmi < 25.0:   return "Normal"
    if bmi < 30.0:   return "Overweight"
    return "Obese"

def _bp_category(sys_bp: float, dia_bp: float) -> str:
    if pd.isna(sys_bp) or pd.isna(dia_bp): return "Unknown"
    if sys_bp < 120 and dia_bp < 80:       return "Normal"
    if sys_bp < 130 and dia_bp < 80:       return "Elevated"
    if sys_bp < 140 or dia_bp < 90:        return "Stage1_HTN"
    return "Stage2_HTN"

def _age_group(age) -> str:
    if pd.isna(age): return "Unknown"
    age = int(age)
    if age < 40:     return "Young"
    if age < 65:     return "Middle"
    if age < 80:     return "Senior"
    return "Elderly"

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    print("\n [FEATURE ENGINEERING] Creando columnas derivadas…")
    df = df.copy()

    # BMI category
    df["bmi_category"] = df["bmi"].apply(_bmi_category)
    print("    bmi_category creada")

    # BP category
    df["bp_category"] = df.apply(
        lambda r: _bp_category(r.get("systolic_bp"), r.get("diastolic_bp")), axis=1
    )
    print("   bp_category creada")

    # Age group
    df["age_group"] = df["age"].apply(_age_group)
    print("   age_group creada")

    # Multimorbidity
    dx_present                 = [c for c in DX_COLUMNS if c in df.columns]
    df["multimorbidity_count"] = df[dx_present].sum(axis=1).astype(int)
    print(f"  multimorbidity_count creada ({len(dx_present)} dx sumadas)")

    # Temperature C
    df["temperature_c"] = ((df["temperature_f"] - 32) * 5 / 9).round(2)
    print("   temperature_c creada (°F → °C)")

    # Risk score
    df["risk_score"] = (df["charlson_index"] * 2 + df["multimorbidity_count"]).astype(int)
    print("  risk_score creada (charlson×2 + multimorbidity_count)")

    # Pulse pressure y MAP
    df["pulse_pressure"] = (df["systolic_bp"] - df["diastolic_bp"]).round(1)
    df["map"]            = (df["diastolic_bp"] + df["pulse_pressure"] / 3).round(1)
    print("   pulse_pressure y map creadas")

    # ─── SÍNDROME METABÓLICO ───────────────────
    metabolic_criteria = (
        df["dx_obesity"].astype(int) +
        df["dx_hypertension"].astype(int) +
        df["dx_type2_diabetes"].astype(int) +
        df["dx_hyperlipidemia"].astype(int) +
        (df["bmi"] >= 30).astype(int)
    )
    df["metabolic_criteria_count"] = metabolic_criteria
    df["metabolic_syndrome"]       = (metabolic_criteria >= 3).astype(int)

    n_metabolic = df["metabolic_syndrome"].sum()
    pct         = round(n_metabolic / len(df) * 100, 1)
    print(f"   metabolic_syndrome creada — {n_metabolic:,} pacientes ({pct}%) cumplen criterios")
    print(f"      Criterios: obesidad + hipertensión + DM2 + hiperlipidemia + BMI≥30 (≥3 = positivo)")
    # ──────────────────────────────────────────

    # Cardiac risk flag
    cardiac_cols            = ["dx_coronary_artery_disease", "dx_heart_failure", "dx_atrial_fibrillation"]
    df["cardiac_risk_flag"] = df[cardiac_cols].max(axis=1).astype(int)
    print("   cardiac_risk_flag creada")

    # Complexity tier
    def complexity(score):
        if score <= 2: return "Low"
        if score <= 5: return "Medium"
        return "High"
    df["complexity_tier"] = df["risk_score"].apply(complexity)
    print("    complexity_tier creada (Low / Medium / High)")

    # Timestamp
    df["etl_timestamp"] = datetime.now(timezone.utc).isoformat()

    print(f"\n   Feature engineering completo — {len(df.columns)} columnas totales")
    return df


#  CARGA

from pymongo import UpdateOne  # agregar este import al inicio del archivo

def load(db, df: pd.DataFrame) -> dict:
    print(f"\n [CARGA] Escribiendo en colección '{TARGET_COLLECTION}' (bulk upsert)…")
    collection = db[TARGET_COLLECTION]
    collection.create_index("patient_id", unique=True)

    records  = df.to_dict(orient="records")
    total    = len(records)
    inserted = 0
    updated  = 0

    for i in range(0, total, CHUNK_SIZE):
        chunk = records[i: i + CHUNK_SIZE]
        
        operations = [
            UpdateOne(
                {"patient_id": doc["patient_id"]},
                {"$set": doc},
                upsert=True
            )
            for doc in chunk
        ]
        
        result = collection.bulk_write(operations, ordered=False)
        inserted += result.upserted_count
        updated  += result.modified_count
        
        done = min(i + CHUNK_SIZE, total)
        print(f"   🔄 {done:,}/{total:,} registros procesados…")

    return {"total": total, "inserted": inserted, "updated": updated}


#  PIPELINE PRINCIPAL

def run_etl():
    print("=" * 55)
    print(" ETL — patients → patients_curated")
    print(f" Inicio: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 55)

    db          = get_database()
    df_raw      = extract(db)
    df_clean    = clean(df_raw)
    df_featured = engineer(df_clean)
    stats       = load(db, df_featured)

    n_flagged   = int(df_featured["data_quality_flag"].sum())
    n_metabolic = int(df_featured["metabolic_syndrome"].sum())
    pct         = round(n_metabolic / stats["total"] * 100, 1)

    print("\n" + "=" * 55)
    print("  ETL COMPLETADO")
    print(f"    Total procesados      : {stats['total']:,}")
    print(f"    Insertados            : {stats['inserted']:,}")
    print(f"     Actualizados          : {stats['updated']:,}")
    print(f"    Filas con flags       : {n_flagged:,}")
    print(f"    Síndrome metabólico   : {n_metabolic:,} pacientes ({pct}%)")
    print(f"   Fin: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 55)


if __name__ == "__main__":
    run_etl()