"""Small query cookbook and safe, isolated CRUD demonstrations."""
from pymongo import ReturnDocument


def cookbook(db):
    return {
        "ver_bahrain": list(db.laps.find({"season": "2025", "gp": "Bahrain Grand Prix", "driver": "VER"}, {"_id": 0, "lap": 1, "laptime": 1}).sort("lap", 1).limit(5)),
        "midlife": list(db.laps.find({"life": {"$gte": 10, "$lt": 20}}, {"_id": 0, "compound": 1, "laptime": 1}).limit(5)),
        "long_soft_stints": list(db.stints.find({"compound": "SOFT", "stint_length": {"$gt": 15}}).limit(5)),
        "sc_2024": db.laps.count_documents({"season": "2024", "is_sc_lap": 1}),
        "multiple_pits": list(db.strategy_decisions.find({"actual_stops": {"$gt": 2}}).limit(5)),
        "dry_compounds": db.laps.count_documents({"compound": {"$in": ["SOFT", "MEDIUM", "HARD"]}}),
        "known_pace": db.laps.count_documents({"fuel_corr": {"$exists": True, "$ne": None}}),
        "slow_laps": list(db.laps.find({"laptime": {"$gt": 120}}).sort("laptime", -1).skip(5).limit(5)),
    }


def delete_matching(collection, query, force=False):
    if not query or set(query) <= {"$where", "$expr"}:
        raise ValueError("An explicit narrow filter is required")
    count = collection.count_documents(query)
    return {"dry_run_count": count, "deleted": collection.delete_many(query).deleted_count if force else 0}


def crud_demo(db):
    # A separate temporary collection keeps the demonstration away from user data.
    from uuid import uuid4
    c = db["crud_demo_"+uuid4().hex]
    try:
        c.insert_one({"_id": "manual", "outcome": "pending"})
        created = c.find_one({"_id": "manual"}) is not None
        c.update_one({"_id": "manual"}, {"$set": {"outcome": "realised"}})
        updated = c.find_one_and_update({"_id": "manual"}, {"$set": {"reviewed": True}},
                                         return_document=ReturnDocument.AFTER)
        c.update_one({"_id": "counter"}, {"$inc": {"runs": 1}, "$setOnInsert": {"kind": "demo"},
                                              "$currentDate": {"updated_at": True}}, upsert=True)
        counter = c.find_one({"_id": "counter"})
        deleted = c.delete_one({"_id": "manual"}).deleted_count == 1
        return {"created": created, "updated": updated["outcome"] == "realised" and updated["reviewed"],
                "deleted": deleted, "counter": counter}
    finally:
        c.drop()
