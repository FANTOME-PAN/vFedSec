import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import flwr as fl
import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch import nn, optim

from core.client_template import TrainClientTemplate
from core.secagg import quantize, reverse_quantize, encrypt
from core.utils import pyobj2bytes, bytes2pyobj
from data_loaders import get_data_loader, get_sample_selector, IDataLoader, ISampleSelector
from models import generate_active_party_local_module, generate_passive_party_local_module
from settings import *

"""=== Client Class ==="""


class TrainActiveParty(TrainClientTemplate):
    """Custom Flower NumPyClient class for training."""

    def __init__(
            self, cid: str, df_pth: Path, client_dir: Path
    ):
        # df = df.sample(frac=1)
        super().__init__(cid, df_pth, client_dir)
        if not self.initialised:
            self.ap_lm: nn.Module = generate_active_party_local_module()
            self.pp_lm_lst: Dict[str, nn.Module] = dict([(key, generate_passive_party_local_module(t))
                                                         for key, t in PASSIVE_PARTY_CIDs.items()])
            self.ap_optimiser = optim.SGD(self.ap_lm.parameters(), lr=LEARNING_RATE, momentum=0.9)
            # self.pp_optimiser = optim.SGD(self.pp_lm.parameters(), lr=LEARNING_RATE, momentum=0.9)
            self.pp_optimisers = dict([(key, optim.SGD(
                lm.parameters(), lr=LEARNING_RATE, momentum=0.9
            )) for key, lm in self.pp_lm_lst.items()])
            self.loader: IDataLoader = None
            self.type2pp_grads: Dict[str, np.ndarray] = dict([(key, np.array(0)) for key in PASSIVE_PARTY_CIDs.keys()])
            self.data = np.array(0)
            self.recv_grad = np.array(0)
            # self.pp_lm_size = sum([o.numel() for o in self.pp_lm.parameters()])
            self.pp_lm_sizes = dict([(key, sum([o.numel() for o in lm.parameters()])) for key, lm in self.pp_lm_lst])

    def setup_round2(self, parameters, config, t: Tuple[List[np.ndarray], int, dict]):
        # load data
        logger.info(f"Client {self.cid}: preprocessing data")
        range_dict = dict([bytes2pyobj(b) for b in parameters[0]])
        self.prf.download(range_dict)
        df = pd.read_csv(self.df_pth)
        self.loader = get_data_loader(df, range_dict)
        # training code
        if ACTIVE_PARTY_PRETRAINED_MODEL_PATH is not None:
            if Path(ACTIVE_PARTY_PRETRAINED_MODEL_PATH).exists():
                self.ap_lm.load_state_dict(torch.load(ACTIVE_PARTY_PRETRAINED_MODEL_PATH))

    def stage0(self, server_rnd, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        # skip the first stage 0
        if server_rnd > 3:
            logger.info(f'Client {self.cid}: updating parameters with received gradients...')
            # need changes
            grad = (parameters[0] + self.partial_grad) & 0xffffffff
            if DEBUG:
                logger.info(f"reconstructed grad = {str(grad)}")
            # grad -= (TARGET_RANGE * (len(self.shared_seed_dict) - 1)) >> 1
            grad = reverse_quantize([grad], CLIP_RANGE, TARGET_RANGE)[0]
            grad -= (len(self.shared_seed_dict) - 1) * CLIP_RANGE

            if DEBUG:
                logger.info(f"aggregate grad = {str(grad)}")
            # assign gradients
            grad = torch.tensor(grad, dtype=torch.float)
            for param in self.pp_lm.parameters():
                _size = param.numel()
                given_grad = grad[:_size].view(param.shape)
                grad = grad[_size:]
                param.grad = given_grad
            # update passive parties' local module
            self.pp_optimiser.step()
            self.pp_optimiser.zero_grad()
            if 'stop' in config:
                torch.save(self.ap_lm, ACTIVE_PARTY_LOCAL_MODULE_SAVE_PATH)
                torch.save(self.pp_lm, PASSIVE_PARTY_LOCAL_MODULE_SAVE_PATH)
                return [], 0, {}

        logger.info('swift client: preparing batch and masked vectors...')
        data, ids, labels = self.loader.next_batch()
        self.data = data
        self.intermediate_output: torch.Tensor = self.ap_lm(data)
        wx = quantize([self.intermediate_output.detach().numpy()], CLIP_RANGE, TARGET_RANGE)[0]
        masked_wx = masking(server_rnd + 1, wx, self.cid, self.shared_seed_dict)
        ret_dict = {}
        for i, (sample_holder_cid, sample_id) in enumerate(ids):
            key = self.shared_secret_dict[sample_holder_cid]
            ret_dict[str(i)] = encrypt(key, sample_id.encode('ascii'))
        return [masked_wx, labels] + \
               [o.detach().numpy() for o in self.pp_lm.parameters()], 0, ret_dict

    def stage2(self, server_rnd, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        self.recv_grad = torch.tensor(parameters[0])
        for party_type in self.type2pp_grads.keys():
            self.type2pp_grads[party_type] = masking(server_rnd, np.zeros(self.pp_lm_sizes[party_type], dtype=int),
                                                     self.cid, self.shared_seed_dict)
        # self.partial_grad = masking(server_rnd, np.zeros(self.pp_lm_size, dtype=int), self.cid, self.shared_seed_dict)
        # update Active Party's local module
        self.intermediate_output.backward(self.recv_grad)
        self.ap_optimiser.step()
        self.ap_optimiser.zero_grad()
        return [], 0, {}


class TrainPassiveParty(TrainClientTemplate):
    """Custom Flower NumPyClient class for training."""

    def __init__(
            self, cid: str, df_pth: Path, client_dir: Path
    ):
        super().__init__(cid, df_pth, client_dir)
        if not self.initialised:
            self.weights = np.array(0)
            self.mask = np.array(0)
            self.selector: ISampleSelector = None
            self.lm = generate_passive_party_local_module()
            self.no_sample_selected = False
            self.output_shape = ()
            self.lm_size = sum([param.numel() for param in self.lm.parameters()])

    def setup_round1(self, parameters, config, t: Tuple[List[np.ndarray], int, dict]):
        df = pd.read_csv(self.df_pth)
        self.selector = get_sample_selector(df)
        with torch.no_grad():
            _input = self.selector.select([df['ID'].values[0]])
            self.output_shape = self.lm(_input).shape[1:]
        t[0].append(np.array([pyobj2bytes((self.cid, self.selector.get_range()))]))
        if DEBUG:
            logger.info(f'client {self.cid}: upload bank list of size {len(t[0][0]) - 1}')

    def stage1(self, server_rnd, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        logger.info(f"Client {self.cid}: reading encrypted batch and computing masked results...")
        for param, given_param in zip(self.lm.parameters(), parameters):
            param.data = torch.tensor(given_param)
        # try read batch
        key = self.shared_secret_dict['0']
        self.mask = torch.zeros(len(config), dtype=bool)
        ids = []
        for str_i, encrypted_id in config.items():
            # logger.info(f'try decrypting {obj_bytes} of type {type(obj_bytes)}')
            i = int(str_i)
            if (sample_id := try_decrypt_and_load(key, encrypted_id)) is not None:
                self.mask[i] = True
                ids += [(i, sample_id.decode('ascii'))]
        ids = sorted(ids, key=lambda x: x[0])
        partial_batch_data = self.selector.select([o[1] for o in ids])
        self.no_sample_selected = True if len(ids) == 0 else False

        # expand the first dimension of the intermediate output to batch_size, padding zeros
        expanded_output = torch.zeros(len(config), *self.output_shape)
        if not self.no_sample_selected:
            self.intermediate_output = self.lm(partial_batch_data)
            expanded_output[self.mask] = self.intermediate_output
        expanded_output = quantize([expanded_output.detach().numpy()], CLIP_RANGE, TARGET_RANGE)[0]
        masked_ret = masking(server_rnd, expanded_output, self.cid, self.shared_seed_dict)
        if DEBUG:
            logger.info(f'client {self.cid}: masking offset = {server_rnd}')
        return [masked_ret], 0, {}

    def stage2(self, server_rnd, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        client_grad = torch.tensor(parameters[0])
        # bank client code
        if not self.no_sample_selected:
            self.intermediate_output.backward(client_grad[self.mask])
            self.partial_grad = []
            for param in self.lm.parameters():
                self.partial_grad += [param.grad.cpu().detach().numpy().flatten()]
            self.partial_grad = np.concatenate(self.partial_grad)
            self.lm.zero_grad()
        else:
            self.partial_grad = np.zeros(self.lm_size)
        # reshape beta to B x 1 from B, then expand it to B x 26
        # do element-wise production between expanded beta and cached_flags (dim: B x 26)
        # finally, sum up along axis 0, get results of dim 26, i.e., the partial agg_grad on this client
        self.partial_grad = quantize([self.partial_grad], CLIP_RANGE, TARGET_RANGE)[0]
        masked_partial_grad = masking(server_rnd, self.partial_grad, self.cid, self.shared_seed_dict)
        if DEBUG:
            logger.info(f'client {self.cid}: uploading masked grad : {masked_partial_grad}')
        else:
            logger.info(f'client {self.cid}: uploading masked gradient')
        return [masked_partial_grad], 0, {}


def train_client_factory(
        cid: str,
        data_path: Path,
        client_dir: Path,
) -> Union[fl.client.Client, fl.client.NumPyClient]:
    """
    Factory function that instantiates and returns a Flower Client for training.
    The federated learning simulation engine will use this function to
    instantiate clients with all necessary dependencies.

    Args:
        cid (str): Identifier for a client node/federation unit. Will be
            constant over the simulation and between train and test stages.
        data_path (Path): Path to CSV data file specific to this client.
        client_dir (Path): Path to a directory specific to this client that is
            available over the simulation. Clients can use this directory for
            saving and reloading client state.

    Returns:
        (Union[Client, NumPyClient]): Instance of Flower Client or NumPyClient.
    """
    if not client_dir.exists():
        os.makedirs(client_dir)
    if cid == "0":
        logger.info("Initializing Active Party for {}", cid)
        return TrainActiveParty(
            cid, df_pth=data_path, client_dir=client_dir
        )
    else:
        logger.info("Initializing Passive Party for {}", cid)
        return TrainPassiveParty(
            cid, df_pth=data_path, client_dir=client_dir
        )
