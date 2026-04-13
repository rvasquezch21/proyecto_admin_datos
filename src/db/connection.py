from pymongo import MongoClient
from dotenv import load_dotenv
import certifi
import os

load_dotenv()

def get_database():
    client = MongoClient(
        os.getenv("MONGO_URI"),
        tlsCAFile=certifi.where(),
        serverSelectionTimeoutMS=60000,  # más tiempo para conectar
        socketTimeoutMS=60000,
        connectTimeoutMS=60000,
        retryWrites=True,
        retryReads=True,
    )
    db = client[os.getenv("MONGO_DB_NAME")]
    print(f"✅ Conexión exitosa a: {db.name}")
    return db