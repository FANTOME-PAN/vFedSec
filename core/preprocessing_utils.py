from typing import Dict, List, Iterable, Tuple, Collection

import numpy as np
import pandas as pd


def vertical_split(df: pd.DataFrame, column_sets: Iterable[List[str]]) \
        -> List[pd.DataFrame]:
    # sanity check
    all_cols = set(df.columns)
    for s in column_sets:
        assert all_cols.issuperset(s)
    # create sub-dataframes
    return [df[cols] for cols in column_sets]


def horizontal_split(df: pd.DataFrame, num_partitions: int, ordered=False) \
        -> List[pd.DataFrame]:
    full_ids = np.arange(len(df)) if ordered else np.random.permutation(len(df))
    ids_lst = np.array_split(full_ids, num_partitions)
    return [df.iloc[ids] for ids in ids_lst]


def categorical_values_to_int(df: pd.DataFrame, columns: Iterable[str],
                              given_dicts: Collection[Dict[any, int]] = ()) \
        -> Tuple[pd.DataFrame, List[Dict[any, int]]]:
    # san check
    assert set(df.columns).issuperset(columns)

    def fn(col):
        cat = dict([(o, i) for i, o in enumerate(set(df[col]))])
        for d in given_dicts:
            if set(d.keys()) == set(cat.keys()):
                cat = d
                break
        df[col] = df[col].apply(lambda x: cat[x])
        return cat

    return df, [fn(c) for c in columns]
