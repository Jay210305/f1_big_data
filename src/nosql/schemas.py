"""Strict validators with nullable analytical measures and typed natural keys."""
NUMBER = {"bsonType": ["double", "int", "long", "decimal", "null"]}
INTEGER = {"bsonType": ["int", "long"]}
TEXT = {"bsonType": "string"}
BASE = {k: TEXT for k in ("_id", "season", "gp", "session", "driver", "compound")}
SCHEMAS = {
    "laps": {"required": ["_id", "season", "gp", "session", "driver", "lap", "stint", "life", "compound"],
             "properties": {**BASE, **{k: INTEGER for k in ("lap", "stint", "life", "_pos", "cluster", "is_sc_lap")},
                            "life": {"bsonType": ["int", "long", "null"]},
                            "stint": {"bsonType": ["int", "long", "null"]},
                            "compound": {"bsonType": ["string", "null"]},
                            **{k: NUMBER for k in ("laptime", "fuel_corr", "pace_above_best", "track_temp_mean")}}},
    "stints": {"required": ["_id", "season", "gp", "session", "driver", "stint"],
               "properties": {**BASE, "stint": {"bsonType": ["int", "long", "null"]},
                              "compound": {"bsonType": ["string", "null"]}, "degradation_slope": NUMBER,
                              "had_cliff": {"bsonType": ["bool", "null"]}}},
    "race_sessions": {"required": ["_id", "season", "gp", "session", "weather", "rcm", "corners", "sc_laps"],
                      "properties": {**BASE, "weather": {"bsonType": "object"},
                                     "corners": {"bsonType": "array", "items": {"bsonType": "object"}},
                                     "rcm": {"bsonType": "array", "items": {"bsonType": "object"}},
                                     "sc_laps": {"bsonType": "array", "items": INTEGER}, "n_laps": INTEGER}},
    "strategy_decisions": {"required": ["_id", "season", "gp", "driver", "scenario", "delta"],
                           "properties": {**BASE, "scenario": TEXT, "delta": NUMBER,
                                          "compounds": {"bsonType": "array", "items": TEXT},
                                          "stop_match": {"bsonType": "bool"}}},
}
SCHEMAS["bench_laps"] = SCHEMAS["laps"]


def validator(name):
    return {"$jsonSchema": {"bsonType": "object", **SCHEMAS[name]}}


def ensure_collections(db):
    existing = set(db.list_collection_names())
    for name in SCHEMAS:
        options = dict(validator=validator(name), validationLevel="strict", validationAction="error")
        if name in existing:
            db.command("collMod", name, **options)
        else:
            db.create_collection(name, **options)
