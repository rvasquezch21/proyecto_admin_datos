import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from connection import get_database

def load_csv_to_mongo(csv_path, collection_name, chunk_size=1000):
    db = get_database()

    if db[collection_name].count_documents({}) > 0:
        print(f"⚠️  La colección '{collection_name}' ya tiene datos. Omitiendo carga.")
        return

    print(f"📂 Leyendo archivo: {csv_path}")
    df = pd.read_csv(csv_path, sep=";")
    total = len(df)
    print(f"📊 {total} registros encontrados con {len(df.columns)} columnas")
    print(f"🔄 Cargando en lotes de {chunk_size}...")

    insertados = 0
    for i in range(0, total, chunk_size):
        chunk = df.iloc[i:i + chunk_size].to_dict(orient="records")
        db[collection_name].insert_many(chunk)
        insertados += len(chunk)
        print(f"   ✅ {insertados}/{total} registros cargados")

    print(f"\n🎉 Carga completa: {insertados} registros en '{collection_name}'")

if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "../data/patients.csv")
    load_csv_to_mongo(csv_path, "patients")