# proyecto_admin_datos
**Administración de Datos — LEAD University**  
Bachillerato en Ingeniería en Ciencia de Datos  
Prof. Alejandro Zamora

Pipeline completo de datos con aplicación de IA sobre un dataset de pacientes médicos.

---

## Estructura del proyecto

```
proyecto_admin_datos/
├── src/
│   ├── data/
│   │   └── patients.csv          # Dataset fuente (Kaggle)
│   ├── db/
│   │   ├── connection.py         # Conexión a MongoDB Atlas
│   │   └── load_data.py          # Carga del CSV a la colección raw
│   ├── etl/
│   │   └── patients.py           # ETL completo (limpieza + feature engineering)
│   └── orchestrate.py            # Script maestro del pipeline
├── pipeline.log                  # Log de ejecuciones (se genera automáticamente)
├── .env                          # Variables de entorno (NO incluido en el repo)
├── .env.example                  # Plantilla de variables de entorno
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Requisitos previos

- Python 3.11 o superior
- Una cuenta en [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) con un cluster creado
- El archivo `patients.csv` colocado en `src/data/`

---

## Instalación

**1. Clonar el repositorio**
```bash
git clone <url-del-repo>
cd proyecto_admin_datos
```

**2. Instalar dependencias**
```bash
pip install -r requirements.txt
```

**3. Configurar variables de entorno**

Copiá el archivo de ejemplo y completá con tus credenciales:
```bash
cp .env.example .env
```

Editá el `.env` con tu connection string de MongoDB Atlas:
```
MONGO_URI=mongodb+srv://usuario:password@cluster.mongodb.net/?retryWrites=true&w=majority
MONGO_DB_NAME=AdminDB
```

> Para obtener tu connection string: Atlas → Connect → Drivers → Python

**4. Agregar tu IP en MongoDB Atlas**

En Atlas → Security → Network Access → Add IP Address → agregar tu IP actual o `0.0.0.0/0` para acceso desde cualquier lugar.

---

## Ejecución

### Opción A — Pipeline completo (recomendado)

Corre todas las etapas en orden desde la raíz del proyecto:

```bash
python3 src/orchestrate.py
```

Esto ejecuta automáticamente:
1. Carga del CSV a MongoDB (`patients`)
2. ETL completo → colección `patients_curated`

El resultado queda registrado en `pipeline.log`.

---

### Opción B — Etapas individuales

**Verificar conexión a MongoDB:**
```bash
cd src/db
python3 connection.py
```

**Cargar datos raw a MongoDB:**
```bash
cd src/db
python3 load_data.py
```

**Correr el ETL:**
```bash
cd src/etl
python3 patients.py
```

---

## Base de datos

| Colección | Descripción |
|---|---|
| `patients` | Datos crudos cargados desde el CSV (100,000 registros, 28 columnas) |
| `patients_curated` | Dataset procesado listo para consumo del modelo IA (43 columnas) |

### Columnas generadas por el ETL

| Columna | Descripción |
|---|---|
| `bmi_category` | Categoría de IMC (Underweight / Normal / Overweight / Obese) |
| `bp_category` | Categoría de presión arterial (Normal / Elevated / Stage1_HTN / Stage2_HTN) |
| `age_group` | Grupo etario (Young / Middle / Senior / Elderly) |
| `multimorbidity_count` | Número total de diagnósticos activos |
| `temperature_c` | Temperatura corporal convertida a Celsius |
| `risk_score` | Puntaje de riesgo (charlson_index × 2 + multimorbidity_count) |
| `pulse_pressure` | Presión diferencial (sistólica − diastólica) |
| `map` | Presión arterial media |
| `metabolic_syndrome` | Flag de síndrome metabólico (≥3 criterios) |
| `cardiac_risk_flag` | Flag de riesgo cardíaco |
| `complexity_tier` | Nivel de complejidad clínica (Low / Medium / High) |
| `data_quality_flag` | Flag de calidad de datos |
| `etl_timestamp` | Timestamp UTC de procesamiento |

---

## Dependencias

```
pymongo
python-dotenv
pandas
numpy
certifi
```

---

## Notas importantes

- El archivo `.env` **nunca** debe subirse al repositorio. Está incluido en `.gitignore`.
- El archivo `patients.csv` también está en `.gitignore` por su tamaño. Cada integrante debe colocarlo manualmente en `src/data/`.
- Si MongoDB rechaza la conexión, verificá que tu IP esté en la whitelist de Network Access en Atlas.
- Si la fecha/hora del sistema está desincronizada, corrés el riesgo de errores SSL. Sincronizá con: `sudo sntp -sS time.apple.com`

---

## Estado del proyecto

- [x] Conexión a base de datos real (MongoDB Atlas)
- [x] Carga de datos raw (100,000 registros)
- [x] ETL completo (limpieza + feature engineering)
- [x] Orquestación con logs por etapas
- [ ] Aplicativo con modelo de IA