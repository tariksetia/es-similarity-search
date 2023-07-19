import json
from typing import Any, Mapping, TypedDict

from pandas import DataFrame, json_normalize


class ESModelNotFound(Exception):
    """raise if ES model is not found"""


class ESResult(TypedDict):
    score: float
    source: Mapping[str, Any]


def get_df(hits: ESResult) -> DataFrame:
    results = json_normalize(hits, sep=".")
    columns = results.columns
    columns = [
        column if "." not in column else column.split(".")[1] for column in columns
    ]
    results.columns = columns
    return results
