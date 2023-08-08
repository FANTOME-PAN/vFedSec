from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import transforms
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
    def next_batch(self) -> Tuple[torch.FloatTensor, List[Tuple[Tuple[str, ...], str]], torch.Tensor]:
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


def get_data_loader() -> IDataLoader:
    return FashionDataLoader(BATCH_SIZE)


def get_sample_selector(client_id: str) -> ISampleSelector:
    return FashionSampleSelector(client_id)


categorical_columns = {'job', 'marital', 'education', 'default', 'housing',
                       'loan', 'contact', 'day', 'month', 'poutcome', 'y'}
target_column = 'y'


class FashionDataLoader(IDataLoader):
    def __init__(self, batch_size):
        self.data: torch.tensor = None
        self.ids: np.ndarray = None
        # self.max_ids = None
        self.type2max_ids: Dict[str, list] = {t: [] for t in INDEX_TO_TYPE}
        self.batch_size = batch_size
        self.train_set = torchvision.datasets.EMNIST(
            "./data",
            split='balanced',
            download=True,
            transform=transforms.Compose([transforms.ToTensor()])
        )
        self.indices = np.random.permutation(len(self.train_set))
        self.all_pp_ids = tuple(
            v[0] for v in PASSIVE_PARTY_CIDs.values()
        )

    # return mini-batch, bathed (client_ID, ID), and labels
    def next_batch(self) -> Tuple[np.ndarray, List[Tuple[Tuple[str, ...], str]], np.ndarray]:
        # List of image-class tuples
        batch: List[Tuple[torch.Tensor, int]] = [
            self.train_set[i] for i in self.indices[:self.batch_size]
        ]
        img = batch[0][0]
        assert img.shape == (1, 28, 28)
        # Retrieve images & targets & indices
        batch_images = torch.stack([img[:, :7, :] for img, _ in batch])
        batch_targets = torch.tensor([gt for _, gt in batch], dtype=int)
        batch_indices = [(self.all_pp_ids, str(idx)) for idx in self.indices[:self.batch_size]]
        # Pop used indices
        self.indices = self.indices[self.batch_size:]
        # Update indices if end
        if len(self.indices) < self.batch_size:
            self.indices = np.random.permutation(len(self.train_set))
        return batch_images, batch_indices, batch_targets


class FashionSampleSelector(ISampleSelector):
    def __init__(self, cid: str):
        self.train_set = torchvision.datasets.EMNIST(
            "./data",
            split='balanced',
            download=True,
            transform=transforms.Compose([transforms.ToTensor()])
        )
        self.cid = cid

    def select(self, ids: List[str]):
        # List of image-class tuples
        batch: List[Tuple[torch.Tensor, int]] = [
            self.train_set[int(i)] for i in ids
        ]
        offset = int(self.cid) * 7
        batch_images = torch.stack([img[:, offset: offset + 7, :] for img, _ in batch])
        return batch_images

    def get_range(self) -> any:
        return 0
