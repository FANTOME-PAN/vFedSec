from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from settings import *


class IDataLoader(ABC):
    @abstractmethod
    def __init__(self, df: pd.DataFrame, range_dict: Dict[str, any]):
        """
        :param df: data held by the active party
        :param range_dict: the dict contains pairs of (passive parties') client IDs and
        their respective sample ranges provided by ISampleSelector.
        For instance, assuming 3 passive parties, their Client IDs '1', '2', '3',
        and their respective sample selectors S1, S2, S3,
        the dict will be
        {
            '1': S1.get_range(),
            '2': S2.get_range(),
            '3': S3.get_range(),
        }
        """
        ...

    @abstractmethod
    def next_batch(self) -> Tuple[torch.FloatTensor, List[Tuple[Tuple[str], str]], torch.Tensor]:
        """
        :rtype: Tuple of 3 elements, i.e., the batched data, the list of (client_IDs, sample_ID), and labels;
        client_IDs should be a tuple containing CIDs of all clients holding the given sample.
        """
        ...


class ISampleSelector(ABC):
    @abstractmethod
    def __init__(self, df: pd.DataFrame):
        """
        :param df: data held by a passive party
        """
        ...

    @abstractmethod
    def select(self, ids: List[str]) -> torch.FloatTensor:
        """
        Select samples, given a list of sample IDs.
        :rtype: the samples, in float tensor form.
        :param ids: the list of sample IDs
        """
        ...

    @abstractmethod
    def get_range(self) -> any:
        """
        Return the range of sample IDs. The range can be any form supported by the Pickle module.
        The results from all passive parties will be gathered and forwarded to the active party by the server.
        """
        ...


def get_data_loader(df: pd.DataFrame, range_dict: Dict[str, any]) -> IDataLoader:
    return ExampleDataLoader(df, range_dict, BATCH_SIZE)


def get_sample_selector(df: pd.DataFrame) -> ISampleSelector:
    return ExampleSampleSelector(df)


categorical_columns = {'Diastolic-blood-pressure', 'Mean-blood-pressure', 'Systolic-blood-pressure',
                       'Heart-Rate','Oxygen-saturation', 'Respiratory-rate'}
target_column = 'y_true'


class ExampleDataLoader(IDataLoader):
    def __init__(self, df: pd.DataFrame, range_dict: Dict[str, any], batch_size):
        self.data: torch.tensor = None
        self.ids: np.ndarray = None
        # self.max_ids = None
        self.type2max_ids: Dict[str, list] = {t: [] for t in INDEX_TO_TYPE}
        self.batch_size = batch_size
        self.ptr = 0
        self.pos_idx = np.array(0)
        self.neg_idx = np.array(0)
        self.new_idx = None
        self.cat_cols = categorical_columns.copy()
        self.set(df, range_dict)

    def id2cid(self, sample_id: str) -> Tuple[str]:
        nid = int(sample_id[1:])
        ret = []
        for t in INDEX_TO_TYPE:
            max_ids = self.type2max_ids[t]
            for cid, max_id in max_ids:
                if nid <= max_id:
                    ret += [cid]
                    break
        if len(ret) == len(INDEX_TO_TYPE):
            return tuple(ret)
        raise RuntimeError(f'{sample_id}: Unexpected ID')

    def set(self, df: pd.DataFrame, range_dict: Dict[str, any]):
        # convert Sample IDs of format 'Nxxxxxx' to integer
        for cid, o in range_dict.items():
            max_id = int(o[1][1:])
            self.type2max_ids[CID_TO_TYPE[cid]].append((cid, max_id))
        for lst in self.type2max_ids.values():
            lst.sort(key=lambda x: x[1])
        # self.max_ids = sorted([(cid, int(o[1][1:])) for cid, o in range_dict.items()], key=lambda x: x[1])

        data: List[torch.Tensor] = []
        self.cat_cols.intersection_update(df.columns)
        for col in df.columns:
            if col in ['ID', target_column]:
                continue
            if col in self.cat_cols:
                # subtract min value from the array, in case some categorical values do not start from 0.
                d_min, d_max = df[col].values.min(), df[col].values.max()
                data += [F.one_hot((torch.tensor(df[col].values) - d_min).to(torch.int64), int(d_max - d_min + 1))]
            else:
                data += [torch.tensor(df[col].values).view(-1, 1)]
        data += [torch.tensor(df[target_column].values).view(-1, 1)]
        data = torch.cat(data, dim=1).float()
        self.ids = df['ID'].values.astype(str)
        self.data = data
        self.shuffle()

    def shuffle(self):
        self.ptr = 0
        self.new_idx = np.random.permutation(self.ids.shape[0])

    # return mini-batch, bathed (client_ID, ID), and labels
    def next_batch(self) -> Tuple[np.ndarray, List[Tuple[str, str]], np.ndarray]:
        batch_data = self.data[self.new_idx][self.ptr: self.ptr + self.batch_size]
        batch_label = batch_data[:, -1].view(-1, 1)
        batch_data = batch_data[:, :-1]
        batch_ids = self.ids[self.new_idx][self.ptr: self.ptr + self.batch_size]
        batch_ids = [(self.id2cid(o), o) for o in batch_ids]
        self.ptr += self.batch_size
        if self.ptr >= self.ids.shape[0]:
            self.shuffle()
        return batch_data, batch_ids, batch_label


class ExampleSampleSelector(ISampleSelector):
    def __init__(self, df: pd.DataFrame):
        cat_cols = categorical_columns.copy()
        cat_cols.intersection_update(df.columns)
        data = []
        for col in df.columns:
            if col == 'ID':
                continue
            if col in cat_cols:
                data += [F.one_hot(torch.tensor(df[col].values).to(torch.int64), int(df[col].values.max() + 1))]
            else:
                data += [torch.tensor(df[col].values).view(-1, 1)]
        self.data = torch.cat(data, dim=1).float()
        self.id2idx = dict(zip(df['ID'], df.index))
        self.min_id, self.max_id = df['ID'].values[0], df['ID'].values[-1]

    def select(self, ids: List[str]):
        return self.data[torch.tensor([self.id2idx[o] for o in ids], dtype=torch.long)]

    def get_range(self) -> any:
        return self.min_id, self.max_id