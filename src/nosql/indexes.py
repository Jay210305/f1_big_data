"""Explicit index evidence without dropping existing user indexes."""
INDEXES = {
    "laps": {"driver_laps": [("season", 1), ("gp", 1), ("driver", 1), ("lap", 1)],
             "compound_life": [("compound", 1), ("life", 1)], "laptime": [("laptime", 1)]},
    "strategy_decisions": {"decision_history": [("season", 1), ("gp", 1), ("driver", 1)]},
    "stints": {"stint_history": [("season", 1), ("gp", 1), ("driver", 1)]},
}


def create_indexes(db):
    return {name: [db[name].create_index(keys, name=label) for label, keys in specs.items()]
            for name, specs in INDEXES.items()}


def explain_query(collection, query, projection=None, hint=None, sort=None):
    command = {"find": collection.name, "filter": query}
    if projection:
        command["projection"] = projection
    if hint is not None:
        command["hint"] = hint
    if sort:
        command["sort"] = dict(sort)
    raw = collection.database.command("explain", command, verbosity="executionStats")
    def stages(node):
        if isinstance(node, dict):
            yield node.get("stage")
            for value in node.values():
                yield from stages(value)
        elif isinstance(node, list):
            for value in node:
                yield from stages(value)
    found = set(stages(raw["queryPlanner"]["winningPlan"]))
    return {**{key: raw["executionStats"][key] for key in
               ("nReturned", "totalDocsExamined", "totalKeysExamined", "executionTimeMillis")},
            "stage": "IXSCAN" if "IXSCAN" in found else "COLLSCAN" if "COLLSCAN" in found else ",".join(sorted(found-{None})),
            "winningPlan": raw["queryPlanner"]["winningPlan"]}


def explain_comparison(db):
    create_indexes(db)
    lap = db.laps.find_one()
    if lap is None:
        raise ValueError("Ingest laps before measuring indexes")
    questions = [
        ("driver history", {k: lap[k] for k in ("season", "gp", "driver")}, "driver_laps"),
        ("season", {"season": lap["season"]}, "driver_laps"),
        ("tyre age", {"compound": "SOFT", "life": {"$gte": 15}}, "compound_life"),
        ("hard tyre", {"compound": "HARD"}, "compound_life"),
        ("slow laps", {"laptime": {"$gt": 120}}, "laptime"),
        ("fast laps", {"laptime": {"$lt": 80}}, "laptime"),
    ]
    return [{"query": label, "filter": query, "before": explain_query(db.laps, query, hint={"$natural": 1}),
             "after": explain_query(db.laps, query, hint=hint)} for label, query, hint in questions]
