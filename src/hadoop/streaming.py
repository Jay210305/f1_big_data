"""Adapt the Python stint job to Hadoop Streaming's stdin/stdout protocol."""
import argparse
import json
import sys
from itertools import groupby

from src.mapreduce.io import canonical
from src.mapreduce.jobs.stint_stats import map_lap, reduce_stint
from src.mapreduce.types import MapContext


def read_pairs(lines):
    for line in lines:
        key, separator, value = line.rstrip("\n").partition("\t")
        if not key or not separator:
            raise ValueError("Expected a Hadoop Streaming key and JSON value separated by a tab")
        yield key, json.loads(value)


def map_lines(lines):
    context = MapContext(task=0, attempt=0)
    for line in lines:
        yield from map_lap(json.loads(line), context)


def reduce_lines(lines):
    for key, group in groupby(read_pairs(lines), key=lambda pair: pair[0]):
        yield from reduce_stint(key, [value for _, value in group])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["map", "reduce"])
    args = parser.parse_args()
    process = map_lines if args.stage == "map" else reduce_lines
    for key, value in process(sys.stdin):
        print(f"{key}\t{canonical(value)}")


if __name__ == "__main__":
    main()
