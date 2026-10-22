"""Deterministic partitioning, independent of Python's randomized hash seed."""
from bisect import bisect_right
from dataclasses import dataclass
from zlib import crc32


@dataclass(frozen=True)
class HashPartitioner:
    def __call__(self, key, num_partitions):
        if num_partitions < 1:
            raise ValueError("num_partitions must be positive")
        return crc32(key.encode("utf-8")) % num_partitions

    def partition_bounds(self):
        return {"method": "crc32", "ranges": None}


@dataclass(frozen=True)
class RangePartitioner:
    boundaries: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "boundaries", tuple(sorted(self.boundaries)))

    def __call__(self, key, num_partitions):
        if num_partitions != len(self.boundaries) + 1:
            raise ValueError("range partitioner needs one fewer boundaries than partitions")
        return bisect_right(self.boundaries, key)

    def partition_bounds(self):
        return {"method": "range", "boundaries": self.boundaries}


@dataclass(frozen=True)
class IdentityPartitioner:
    def __call__(self, key, num_partitions):
        if num_partitions < 1:
            raise ValueError("num_partitions must be positive")
        return 0

    def partition_bounds(self):
        return {"method": "identity", "ranges": ["all keys -> 0"]}
