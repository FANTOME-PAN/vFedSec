from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from settings import BATCH_SIZE


class IDataLoader(ABC):
    @abstractmethod
    def __init__(self, df: pd.DataFrame, range_dict: Dict[str, any]):
        """
        :param df: data held by the active party
        :param range_dict: the dict contains pairs of (passive parties') client IDs and
        their respective sample ranges provided by ISampleSelector.
        For instance, assuming 3 passive parties, their Client IDs 'p1', 'p2', 'p3',
        and their respective sample selectors S1, S2, S3,
        the dict will be
        {
            'p1': S1.get_range(),
            'p2': S2.get_range(),
            'p3': S3.get_range(),
        }
        """
        ...

    @abstractmethod
    def next_batch(self) -> Tuple[torch.FloatTensor, List[Tuple[str, str]], torch.Tensor]:
        """
        :rtype: Tuple of 3 elements, i.e., the batched data, the list of (client_ID, sample_ID), and labels
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


class ExampleDataLoader(IDataLoader):
    def __init__(self, df: pd.DataFrame, range_dict: Dict[str, any], batch_size):
        self.data: torch.tensor = None
        self.ids: np.ndarray = None
        self.max_ids = None
        self.batch_size = batch_size
        self.ptr = 0
        self.pos_idx = np.array(0)
        self.neg_idx = np.array(0)
        self.new_idx = None
        self.cat_cols = ('default', 'housing', 'loan', 'contact', 'day', 'month', 'poutcome')
        self.set(df, range_dict)

    def id2cid(self, sample_id: str) -> str:
        nid = int(sample_id[1:])
        for cid, max_id in self.max_ids:
            if nid <= max_id:
                return cid
        raise RuntimeError(f'{sample_id}: Unexpected ID')

    def set(self, df: pd.DataFrame, range_dict: Dict[str, any]):
        print(range_dict)
        # convert Sample IDs of format 'Nxxxxxx' to integer
        self.max_ids = sorted([(cid, int(o[1][1:])) for cid, o in range_dict.items()], key=lambda x: x[1])
        data = []
        for col in df.columns:
            if col == 'ID':
                continue
            if col in self.cat_cols:
                data += [F.one_hot(torch.tensor(df[col].values), df[col].values.max() + 1)]
            else:
                data += [torch.tensor(df[col].values).view(-1, 1)]
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
        cat_col = ['job', 'marital', 'education']
        data = []
        for col in df.columns:
            if col == 'ID':
                continue
            if col in cat_col:
                data += [F.one_hot(torch.tensor(df[col].values), df[col].values.max() + 1)]
            else:
                data += [torch.tensor(df[col].values).view(-1, 1)]
        self.data = torch.cat(data, dim=1).float()
        self.id2idx = dict(zip(df['ID'], df.index))
        self.min_id, self.max_id = df['ID'].values[0], df['ID'].values[-1]

    def select(self, ids: List[str]):
        return self.data[torch.tensor([self.id2idx[o] for o in ids], dtype=torch.long)]

    def get_range(self) -> any:
        return self.min_id, self.max_id
