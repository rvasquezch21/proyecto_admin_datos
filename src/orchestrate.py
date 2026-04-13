import subprocess
import sys
import os
from datetime import datetime, timezone

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline.log")

def log(msg: str):
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp} UTC] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def run_step(name: str, script_path: str) -> bool:
    log(f"▶ INICIANDO: {name}")
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False
    )
    if result.returncode == 0:
        log(f"✅ COMPLETADO: {name}")
        return True
    else:
        log(f"❌ ERROR en: {name} (código {result.returncode})")
        return False

def run_pipeline():
    base = os.path.dirname(os.path.abspath(__file__))

    steps = [
        ("Carga de datos raw → MongoDB",  os.path.join(base, "db",  "load_data.py")),
        ("ETL → patients_curated",         os.path.join(base, "etl", "patients.py")),
    ]

    log("=" * 55)
    log("PIPELINE INICIADO")
    log("=" * 55)

    for name, path in steps:
        success = run_step(name, path)
        if not success:
            log("⛔ Pipeline detenido por error.")
            sys.exit(1)

    log("=" * 55)
    log("PIPELINE COMPLETADO EXITOSAMENTE")
    log("=" * 55)

if __name__ == "__main__":
    run_pipeline()