"""Per-attempt stage traces; failed attempts never enter the committed trace."""
from collections import Counter
from pathlib import Path

from .io import canonical


class TraceWriter:
    def __init__(self, path, enabled, sample=None):
        self.path = Path(path)
        self.enabled = enabled
        self.sample = sample
        self.counts = Counter()
        self.handles = {}

    def emit(self, stage, task, partition, key, value):
        self.counts[stage] += 1
        if not self.enabled or (self.sample is not None and self.counts[stage] > self.sample):
            return
        if stage not in self.handles:
            self.handles[stage] = (self.path / f"{stage}.jsonl").open("w", encoding="utf-8", newline="\n")
        self.handles[stage].write(canonical(dict(stage=stage, task=task,
            partition=partition, key=key, value=value)) + "\n")

    def close(self):
        for handle in self.handles.values():
            handle.close()
