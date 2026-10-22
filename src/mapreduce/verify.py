"""Trace-only integrity checks and independently supplied reference reductions."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from .io import canonical, read_jsonl


def verify_trace(path, reference, *, combiner=None):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["sampled"]:
        raise ValueError("sampled traces cannot prove complete coverage")
    stages = {name: list(read_jsonl(path / f"{name}.jsonl"))
              for name in ("map", "combine", "shuffle", "reduce")}
    assert len(stages["map"]) == manifest["kv_before_combine"]
    assert len(stages["shuffle"]) == manifest["kv_after_combine"]
    num_partitions = manifest["reduce_tasks"]
    for name, rows in stages.items():
        for row in rows:
            partition = row["partition"]
            if isinstance(partition, bool) or not isinstance(partition, int):
                raise ValueError(f"{name} row has a non-integer partition: {partition!r}")
            if not 0 <= partition < num_partitions:
                raise ValueError(f"{name} row partition {partition} is outside the "
                                 f"manifest range [0, {num_partitions - 1}]")
    for name in ("shuffle", "reduce"):
        for row in stages[name]:
            if row["task"] != row["partition"]:
                raise ValueError(f"{name} row task {row['task']} does not match its "
                                 f"reducer partition {row['partition']}")
    def multiset(rows):
        return Counter((r["partition"], r["key"], canonical(r["value"])) for r in rows)
    if manifest["combiner"]:
        if combiner is None:
            raise ValueError("combiner reference required to verify compressed values")
        groups = defaultdict(list)
        for row in stages["map"]:
            groups[(row["task"], row["partition"], row["key"])].append(row["value"])
        expected = []
        for (task, partition, key), values in groups.items():
            for k, value in combiner(key, values):
                expected.append(dict(task=task, partition=partition, key=k, value=value))
        assert multiset(expected) == multiset(stages["combine"]), "combiner lost/duplicated values"
        assert multiset(stages["combine"]) == multiset(stages["shuffle"])
    else:
        assert multiset(stages["map"]) == multiset(stages["shuffle"]), "shuffle lost/duplicated values"
    partitions, grouped = defaultdict(set), defaultdict(list)
    for row in stages["shuffle"]:
        partitions[row["key"]].add(row["partition"])
        grouped[row["key"]].append(row["value"])
    assert all(len(p) == 1 for p in partitions.values()), "key reached multiple reducers"
    expected = sorted((pair for key, values in grouped.items() for pair in reference(key, values)),
                      key=lambda pair: pair[0])
    actual = sorted(([r["key"], r["value"]] for r in stages["reduce"]), key=lambda pair: pair[0])
    def equivalent(a, b):
        if isinstance(a, dict) and isinstance(b, dict):
            return a.keys() == b.keys() and all(equivalent(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            return len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return math.isclose(a, b, abs_tol=1e-9, rel_tol=1e-9)
        return a == b
    assert equivalent(expected, actual), "independent reduction disagrees"
    return dict(verified=True, input_records=manifest["input_records"],
                map_pairs=len(stages["map"]), keys=len(grouped),
                nonempty_partitions=len({r["partition"] for r in stages["shuffle"]}))
