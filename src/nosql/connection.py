"""Short-timeout connections; Mongo remains an optional operational store."""
import os
from pymongo import MongoClient


def get_database(uri=None, db_name=None):
    client = MongoClient(uri or os.getenv("MONGO_URI", "mongodb://localhost:27017"),
                         serverSelectionTimeoutMS=2000, connectTimeoutMS=2000)
    try:
        client.admin.command("ping")
    except Exception:
        client.close()
        raise
    return client[db_name or os.getenv("MONGO_DB", "f1_tyre")]


connect = get_database
