from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

def get_database():
    client = MongoClient(os.getenv("MONGO_URI"))
    db = client[os.getenv("MONGO_DB_NAME")]
    print(f"✅ Conexión exitosa a: {db.name}")
    return db

if __name__ == "__main__":
    get_database()