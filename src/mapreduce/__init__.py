"""Local, disk-backed MapReduce with spawn workers and auditable stage traces."""
from .types import JobSpec
from .runner import LocalJobRunner, run_job

__all__ = ["JobSpec", "LocalJobRunner", "run_job"]
