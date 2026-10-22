"""Thin read API shared by optimizer and the history dashboard."""
import math
from .connection import get_database


def known_compounds(documents):
    """The compounds a driver actually ran, sorted.

    The stored history keeps a canonical BSON null for a compound the source
    could not resolve. An unknown tyre is not part of the compound set, so it
    is dropped here instead of reaching a comparison with the known ones.
    """
    return sorted({doc.get("compound") for doc in documents
                   if doc.get("compound") is not None})


class Repository:
    def __init__(self, db=None, *, uri=None, db_name=None):
        self.db = db if db is not None else get_database(uri, db_name)

    def lap_history(self, season, gp, session, driver):
        """The driver's laps, oldest first.

        A null compound is returned verbatim: the caller decides how an
        unknown tyre is treated (see known_compounds).
        """
        return list(self.db.laps.find(dict(season=str(season), gp=gp, session=session, driver=driver),
                                      {"_id": 0}).sort("lap", 1))

    def compounds_used(self, season, gp, session, driver, lap=None):
        """The compounds already used, ignoring the unknown (null) ones."""
        documents = self.lap_history(season, gp, session, driver)
        if lap is not None:
            documents = [doc for doc in documents
                         if doc.get("lap") is not None and doc["lap"] <= lap]
        return known_compounds(documents)

    def stint_for(self, season, gp, session, driver, lap):
        return self.db.laps.find_one(dict(season=str(season), gp=gp, session=session, driver=driver,
                                          lap={"$lte": lap}), {"_id": 0}, sort=[("lap", -1)])

    def session_lap_count(self, season, gp, session):
        doc = self.db.race_sessions.find_one({"_id": f"{season}|{gp}|{session}"}, {"n_laps": 1})
        if doc is None:
            raise LookupError(f"Session missing from Mongo: {season}|{gp}|{session}")
        return int(doc["n_laps"])

    def latest_decisions(self, gp, driver):
        return self.decisions(gp=gp, driver=driver)

    @staticmethod
    def decision_filter(season=None, gp=None, driver=None, compound=None):
        return {k: str(v) if k == "season" else v for k, v in
                (("season", season), ("gp", gp), ("driver", driver), ("compounds", compound)) if v is not None}

    def decisions(self, season=None, gp=None, driver=None, compound=None, page=0, page_size=25):
        if page < 0 or not 1 <= page_size <= 1000:
            raise ValueError("page >= 0 and 1 <= page_size <= 1000 required")
        return list(self.db.strategy_decisions.find(self.decision_filter(season, gp, driver, compound),
                    {"_id": 0}).sort([("season", -1), ("gp", 1), ("driver", 1), ("scenario", 1)])
                    .skip(page * page_size).limit(page_size))

    def decision_count(self, **filters):
        return self.db.strategy_decisions.count_documents(self.decision_filter(**filters))

    def delta_histogram(self, season=None, gp=None, driver=None, compound=None, width=20):
        if not math.isfinite(width) or width <= 0:
            raise ValueError("positive finite histogram width required")
        return list(self.db.strategy_decisions.aggregate([
            {"$match": {**self.decision_filter(season, gp, driver, compound), "delta": {"$type": "number"}}},
            {"$group": {"_id": {"$multiply": [{"$floor": {"$divide": ["$delta", width]}}, width]},
                         "count": {"$sum": 1}}},
            {"$project": {"_id": 0, "bin_start": "$_id", "bin_end": {"$add": ["$_id", width]}, "count": 1}},
            {"$sort": {"bin_start": 1}}]))

    def collection_stats(self, collection="strategy_decisions"):
        stats = self.db.command("collStats", collection)
        return {key: stats.get(key) for key in ("count", "size", "storageSize", "totalIndexSize", "nindexes")}
