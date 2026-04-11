
import sys, os, pandas as pd, numpy as np
from datetime import datetime, timezone

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "db"))
from connection import get_database

SOURCE_COLLECTION = "patients"
TARGET_COLLECTION = "patients_curated"
CHUNK_SIZE = 1000

DX_COLUMNS = ["dx_hypertension","dx_type2_diabetes","dx_hyperlipidemia","dx_obesity",
    "dx_coronary_artery_disease","dx_heart_failure","dx_atrial_fibrillation",
    "dx_chronic_kidney_disease","dx_copd","dx_asthma","dx_depression","dx_anxiety",
    "dx_hypothyroidism","dx_osteoarthritis","dx_type1_diabetes"]
NUMERIC_COLS = ["age","bmi","systolic_bp","diastolic_bp","heart_rate","temperature_f","charlson_index"]
CATEGORICAL_COLS = ["sex","smoking_status","alcohol_use","exercise_level","insurance_type"]
CLINICAL_RANGES = {"age":(0,120),"bmi":(10,70),"systolic_bp":(50,300),
    "diastolic_bp":(30,200),"heart_rate":(20,250),"temperature_f":(90,110)}

def extract(db):
    print("\n[EXTRACCIÓN] Leyendo colección:", SOURCE_COLLECTION)
    records = list(db[SOURCE_COLLECTION].find({}, {"_id": 0}))
    if not records: raise ValueError("La colección está vacía.")
    df = pd.DataFrame(records)
    print(f"    {len(df):,} documentos extraídos — {len(df.columns)} columnas")
    return df

def clean(df):
    print("\n [LIMPIEZA] Iniciando transformaciones...")
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates(subset="patient_id", keep="first")
    dropped = before - len(df)
    if dropped: print(f"   {dropped} duplicados eliminados")
    for col in NUMERIC_COLS:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
    df["age"] = df["age"].astype("Int64")
    for col in DX_COLUMNS:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in NUMERIC_COLS:
        if col in df.columns:
            n = df[col].isna().sum()
            if n:
                m = df[col].median()
                df[col] = df[col].fillna(m)
                print(f"    {col}: {n} nulos → mediana ({m:.2f})")
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            n = df[col].isna().sum()
            if n: df[col] = df[col].fillna("unknown")
    inconsistent = df["diastolic_bp"] >= df["systolic_bp"]
    if inconsistent.sum(): print(f"   {inconsistent.sum()} pacientes con diastólica >= sistólica")
    if "dx_type1_diabetes" in df.columns and "dx_type2_diabetes" in df.columns:
        both = (df["dx_type1_diabetes"]==1) & (df["dx_type2_diabetes"]==1)
        if both.sum(): print(f"    {both.sum()} pacientes con DM1 y DM2 simultáneas")
    df["data_quality_flag"] = False
    df["data_quality_issues"] = ""
    for col,(lo,hi) in CLINICAL_RANGES.items():
        if col not in df.columns: continue
        oor = (pd.to_numeric(df[col],errors="coerce") < lo) | (pd.to_numeric(df[col],errors="coerce") > hi)
        if oor.sum():
            df.loc[oor,"data_quality_flag"] = True
            df.loc[oor,"data_quality_issues"] += f"{col}_out_of_range; "
            print(f"   {col}: {oor.sum()} valores fuera de [{lo},{hi}]")
    print(f"    Limpieza completa — {len(df):,} registros")
    return df

def engineer(df):
    print("\n [FEATURE ENGINEERING] Creando columnas derivadas...")
    df = df.copy()
    df["bmi_category"] = df["bmi"].apply(lambda b: "Unknown" if pd.isna(b) else "Underweight" if b<18.5 else "Normal" if b<25 else "Overweight" if b<30 else "Obese")
    df["bp_category"] = df.apply(lambda r: "Unknown" if pd.isna(r.get("systolic_bp")) else "Normal" if r["systolic_bp"]<120 and r["diastolic_bp"]<80 else "Elevated" if r["systolic_bp"]<130 and r["diastolic_bp"]<80 else "Stage1_HTN" if r["systolic_bp"]<140 or r["diastolic_bp"]<90 else "Stage2_HTN", axis=1)
    df["age_group"] = df["age"].apply(lambda a: "Unknown" if pd.isna(a) else "Young" if int(a)<40 else "Middle" if int(a)<65 else "Senior" if int(a)<80 else "Elderly")
    dx = [c for c in DX_COLUMNS if c in df.columns]
    df["multimorbidity_count"] = df[dx].sum(axis=1).astype(int)
    df["temperature_c"] = ((df["temperature_f"]-32)*5/9).round(2)
    df["risk_score"] = (df["charlson_index"]*2 + df["multimorbidity_count"]).astype(int)
    df["pulse_pressure"] = (df["systolic_bp"]-df["diastolic_bp"]).round(1)
    df["map"] = (df["diastolic_bp"]+df["pulse_pressure"]/3).round(1)
    met = (df["dx_obesity"].astype(int)+df["dx_hypertension"].astype(int)+
           df["dx_type2_diabetes"].astype(int)+df["dx_hyperlipidemia"].astype(int)+(df["bmi"]>=30).astype(int))
    df["metabolic_criteria_count"] = met
    df["metabolic_syndrome"] = (met>=3).astype(int)
    n_met = df["metabolic_syndrome"].sum()
    print(f"    metabolic_syndrome — {n_met:,} pacientes ({round(n_met/len(df)*100,1)}%) cumplen criterios")
    df["cardiac_risk_flag"] = df[["dx_coronary_artery_disease","dx_heart_failure","dx_atrial_fibrillation"]].max(axis=1).astype(int)
    df["complexity_tier"] = df["risk_score"].apply(lambda s: "Low" if s<=2 else "Medium" if s<=5 else "High")
    df["etl_timestamp"] = datetime.now(timezone.utc).isoformat()
    print(f"   Feature engineering completo — {len(df.columns)} columnas totales")
    return df

def load(db, df):
    print(f"\n[CARGA] Escribiendo en {TARGET_COLLECTION} (upsert)...")
    col = db[TARGET_COLLECTION]
    col.create_index("patient_id", unique=True)
    records = df.to_dict(orient="records")
    total = len(records); inserted = 0; updated = 0
    for i in range(0, total, CHUNK_SIZE):
        for doc in records[i:i+CHUNK_SIZE]:
            r = col.update_one({"patient_id":doc["patient_id"]},{"$set":doc},upsert=True)
            if r.upserted_id: inserted += 1
            else: updated += 1
        print(f"    {min(i+CHUNK_SIZE,total):,}/{total:,} procesados...")
    return {"total":total,"inserted":inserted,"updated":updated}

def run_etl():
    print("="*55)
    print("  ETL — patients → patients_curated")
    print(f"  Inicio: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("="*55)
    db = get_database()
    df_raw = extract(db)
    df_clean = clean(df_raw)
    df_feat = engineer(df_clean)
    stats = load(db, df_feat)
    n_flag = int(df_feat["data_quality_flag"].sum())
    n_met = int(df_feat["metabolic_syndrome"].sum())
    print("\n"+"="*55)
    print("  ETL COMPLETADO")
    print(f"    Total procesados    : {stats['total']:,}")
    print(f"    Insertados          : {stats['inserted']:,}")
    print(f"     Actualizados        : {stats['updated']:,}")
    print(f"    Filas con flags     : {n_flag:,}")
    print(f"    Síndrome metabólico : {n_met:,} ({round(n_met/stats['total']*100,1)}%)")
    print("="*55)

if __name__ == "__main__":
    run_etl()
