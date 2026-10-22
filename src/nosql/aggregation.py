"""Server-native reductions; stint filtering exactly follows fit_degradation."""
KEYS = ["season", "gp", "session", "driver", "stint"]
COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]


def compound_pace(match=None):
    return [{"$match": {"laptime": {"$type": "number"}, **(match or {})}},
            {"$group": {"_id": {"season": "$season", "compound": "$compound"},
                         "mean_laptime": {"$avg": "$laptime"}, "std_laptime": {"$stdDevSamp": "$laptime"},
                         "n_laps": {"$sum": 1}}}, {"$sort": {"_id.season": 1, "_id.compound": 1}}]


def stint_slopes(match=None):
    # _pos is assigned in life order BEFORE the SC filter at ingestion. Centered
    # second pass avoids catastrophic cancellation in sum(x*y)-sum(x)*sum(y)/n.
    return [{"$match": {"compound": {"$in": COMPOUNDS}, "_pos": {"$gte": 3}, "is_sc_lap": 0,
                        **(match or {})}},
            {"$sort": {**{k: 1 for k in KEYS}, "life": 1}},
            {"$group": {"_id": {k: "$"+k for k in KEYS}, "n_laps": {"$sum": 1},
                         "points": {"$push": {"x": "$life", "y": "$fuel_corr"}},
                         "compound": {"$first": "$compound"}, "best_pace": {"$min": "$fuel_corr"},
                         "mean_dirty_air_frac": {"$avg": "$dirty_air_frac"},
                         "track_temp_mean": {"$avg": "$track_temp_mean"}}},
            {"$match": {"n_laps": {"$gte": 4}}},
            {"$set": {"xpoints": {"$filter": {"input": "$points", "as": "p", "cond": {"$isNumber": "$$p.x"}}},
                      "points": {"$filter": {"input": "$points", "as": "p", "cond": {"$and": [
                          {"$isNumber": "$$p.x"}, {"$isNumber": "$$p.y"}]}}}}},
            {"$set": {"xmean": {"$avg": "$xpoints.x"}, "pair_xmean": {"$avg": "$points.x"},
                      "ymean": {"$avg": "$points.y"}, "nx": {"$size": "$xpoints"}, "np": {"$size": "$points"}}},
            {"$set": {"sxx": {"$sum": {"$map": {"input": "$xpoints", "as": "p", "in":
                          {"$pow": [{"$subtract": ["$$p.x", "$xmean"]}, 2]}}}},
                      "sxy": {"$sum": {"$map": {"input": "$points", "as": "p", "in":
                          {"$multiply": [{"$subtract": ["$$p.x", "$pair_xmean"]},
                                         {"$subtract": ["$$p.y", "$ymean"]}]}}}}}},
            {"$match": {"sxx": {"$gt": 0}, "nx": {"$gt": 1}, "np": {"$gt": 1}}},
            {"$project": {"_id": 0, "key": {"$concat": ["$_id.season", "|", "$_id.gp", "|",
                         "$_id.session", "|", "$_id.driver", "|", {"$toString": "$_id.stint"}]},
                          **{k: "$_id."+k for k in KEYS}, "compound": 1, "n_laps": 1,
                          "beta": {"$multiply": [{"$divide": ["$sxy", "$sxx"]},
                              {"$divide": [{"$subtract": ["$nx", 1]}, {"$subtract": ["$np", 1]}]}]}, "best_pace": 1,
                          "mean_dirty_air_frac": 1, "track_temp_mean": 1}}, {"$sort": {"key": 1}}]


def safety_car_share(match=None):
    return [{"$match": match or {}}, {"$group": {"_id": {"season": "$season", "gp": "$gp"},
            "n_laps": {"$sum": 1}, "sc_share": {"$avg": "$is_sc_lap"}}}, {"$sort": {"_id.season": 1}}]


def race_control_counts(match=None):
    return [{"$match": match or {}}, {"$unwind": "$rcm"},
            {"$match": {"rcm.cat": {"$in": ["SafetyCar", "VirtualSafetyCar", "VSC"]}}},
            {"$group": {"_id": {"season": "$season", "gp": "$gp"}, "messages": {"$sum": 1}}}]


def pace_facet(match=None):
    return [{"$match": match or {}}, {"$facet": {
        "worst_laps": [{"$sort": {"laptime": -1}}, {"$limit": 5}],
        "compound_pace": [{"$group": {"_id": "$compound", "mean_laptime": {"$avg": "$laptime"}}}]}}]


# Explicit public alias used by the cross-engine evidence command.
stint_pipeline = stint_slopes
