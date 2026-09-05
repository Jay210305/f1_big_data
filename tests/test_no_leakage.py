"""Leakage-critical assertion: no cliff-derived feature may enter X."""

from src.models.rul_data import CLIFF_DERIVED, feature_cols


def test_no_cliff_derived_features_in_X():
    """The feature selector must strip every cliff-derived column (spec 2.1).

    Building a frame that contains BOTH clean telemetry and every cliff-derived
    column, then asserting the intersection is empty, guards against regressions
    that re-introduce label leakage into the model inputs.
    """
    # exercise the real selector against a frame carrying all cliff features
    df = {}
    for c in CLIFF_DERIVED:
        df[c] = [0.0]
    df["speed_mean"] = [300.0]
    df["compound"] = ["SOFT"]
    df["session_type"] = ["Race"]
    df.update({"season": ["2025"], "gp": ["T"], "session": ["Race"],
               "lap": [1], "stint": [1]})

    import polars as pl
    frame = pl.DataFrame(df)
    feats = feature_cols(frame)
    assert CLIFF_DERIVED & set(feats) == set(), \
        f"cliff-derived leaked into X: {CLIFF_DERIVED & set(feats)}"


def test_cliff_derived_not_in_identifiers_or_labels():
    from src.models.rul_data import EXCLUDED
    for c in CLIFF_DERIVED:
        assert c in EXCLUDED, f"{c} missing from EXCLUDED"
