"""Pickleable job contracts; mappers may emit zero, one, or many pairs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .partitioner import HashPartitioner

KV = tuple[str, Any]


@dataclass(frozen=True)
class MapContext:
    task: int
    attempt: int


class Mapper(Protocol):
    def __call__(self, record: dict, ctx: MapContext) -> list[KV]: ...


class Reducer(Protocol):
    def __call__(self, key: str, values: list[Any]) -> list[KV]: ...


class Combiner(Reducer, Protocol):
    """Associative local reduction, with the same value type as the mapper."""


class Partitioner(Protocol):
    def __call__(self, key: str, num_partitions: int) -> int: ...


@dataclass
class JobSpec:
    name: str
    mapper: Mapper
    reducer: Reducer
    combiner: Combiner | None = None
    partitioner: Partitioner = field(default_factory=HashPartitioner)
    num_partitions: int = 4
    input_splitter: Callable | None = None
    output_sink: Callable | None = None
    max_attempts: int = 3
