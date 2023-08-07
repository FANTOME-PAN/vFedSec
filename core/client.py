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
from core.secagg import quantize, reverse_quantize, encrypt, decrypt
from core.utils import pyobj2bytes, bytes2pyobj, pack_local_modules
from data_loaders import get_data_loader, get_sample_selector, IDataLoader, ISampleSelector
from models import generate_active_party_local_module, generate_passive_party_local_module
from settings import *


def rng_masking(x: np.ndarray, cid, rng_dict: Dict[str, np.random.RandomState], target_range=(1 << 32)) -> np.ndarray:
    for other_cid, rng in rng_dict.items():
        sign = 1 if cid < other_cid else -1
        x += sign * rng.randint(0, target_range, x.shape, dtype=np.int64)
    return x


def try_decrypt_and_load(key, ciphertext: bytes) -> Union[object, None]:
    try:
        plaintext = decrypt(key, ciphertext)
        ret = plaintext
        return ret
    except:
        return None


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
            self.pp_lm_dict: Dict[str, nn.Module] = {t: generate_passive_party_local_module(t)
                                                     for t in INDEX_TO_TYPE}
            self.ap_optimiser = optim.SGD(self.ap_lm.parameters(), lr=LEARNING_RATE, momentum=0.9)
            self.pp_optimisers = {t: optim.SGD(lm.parameters(), lr=LEARNING_RATE, momentum=0.9)
                                  for t, lm in self.pp_lm_dict.items()}
            self.loader: IDataLoader = None
            self.type2pp_grads: Dict[str, np.ndarray] = {key: np.array(0) for key in PASSIVE_PARTY_CIDs.keys()}
            self.data = np.array(0)
            self.recv_grad = np.array(0)
            self.pp_lm_sizes = {t: sum([o.numel() for o in lm.parameters()]) for t, lm in self.pp_lm_dict.items()}

    def setup_round2(self, parameters, config, t: Tuple[List[np.ndarray], int, dict]):
        # load data
        if ENABLE_PROFILER:
            self.prf.tic()
        self.bwd_rng_dict = {}
        logger.info(f"Client {self.cid}: preprocessing data")
        # range_dict = dict([bytes2pyobj(b) for b in parameters[0]])
        # df = pd.read_csv(self.df_pth)
        self.loader = get_data_loader()
        # training code
        if ACTIVE_PARTY_PRETRAINED_MODEL_PATH is not None:
            if Path(ACTIVE_PARTY_PRETRAINED_MODEL_PATH).exists():
                self.ap_lm.load_state_dict(torch.load(ACTIVE_PARTY_PRETRAINED_MODEL_PATH))
        if ENABLE_PROFILER:
            self.prf.toc(is_overhead=False)
            # self.prf.download(range_dict, range_dict)

    def stage0(self, server_rnd, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        # skip the first stage 0
        if server_rnd > 3:
            logger.info(f'Client {self.cid}: updating parameters with received gradients...')
            for idx, party_type in enumerate(INDEX_TO_TYPE):
                grad = self.type2pp_grads[party_type]
                grad = (grad + parameters[idx]) & 0xffffffff
                grad = reverse_quantize([grad], CLIP_RANGE, TARGET_RANGE)[0]
                grad -= (len(PASSIVE_PARTY_CIDs[party_type]) - 1) * CLIP_RANGE
                self.type2pp_grads[party_type] = grad

            for party_type, grad in self.type2pp_grads.items():
                if len(PASSIVE_PARTY_CIDs[party_type]) == 1:
                    logger.info(f"Client {self.cid}: skip passive party type {party_type} "
                                f"- passive cluster contains only one client")
                    continue
                # assign gradients
                if VALIDATION:
                    logger.info(f"VALIDATION GRAD: {grad[:10]}")
                    continue
                grad = torch.tensor(grad, dtype=torch.float)
                if VALIDATION:
                    logger.info(f"VALIDATION GRAD: {grad[:10]}")
                    continue
                pp_lm = self.pp_lm_dict[party_type]
                optimiser = self.pp_optimisers[party_type]
                for param in pp_lm.parameters():
                    _size = param.numel()
                    given_grad = grad[:_size].view(param.shape)
                    grad = grad[_size:]
                    param.grad = given_grad
                # update passive parties' local module
                optimiser.step()
                optimiser.zero_grad()
            if 'stop' in config:
                torch.save(self.ap_lm, ACTIVE_PARTY_LOCAL_MODULE_SAVE_PATH)
                for party_type, pp_lm in self.pp_lm_dict.items():
                    torch.save(pp_lm, PASSIVE_PARTY_LOCAL_MODULE_SAVE_PATH_FORMAT % party_type)
                return [], 0, {}

        logger.info('swift client: preparing batch and masked vectors...')
        if ENABLE_PROFILER:
            self.prf.tic()
        data, ids, labels = self.loader.next_batch()
        self.data = data
        self.intermediate_output: torch.Tensor = self.ap_lm(data)
        if ENABLE_PROFILER:
            self.prf.toc(is_overhead=False)
            self.prf.tic()
        wx = self.intermediate_output.detach().numpy()
        if VALIDATION:
            wx = np.zeros_like(wx)
        wx = quantize([wx], CLIP_RANGE, TARGET_RANGE)[0]
        masked_wx = rng_masking(wx, self.cid, self.fwd_rng_dict)
        ret_dict = {t: [] for t in INDEX_TO_TYPE}
        for cids, sample_id in ids:
            for cid in cids:
                if ENABLE_PROFILER:
                    self.prf.toc(is_overhead=False)
                    self.prf.tic()
                key = self.shared_secret_dict[cid]
                encrypted_sid = encrypt(key, sample_id.encode('ascii'))
                if ENABLE_PROFILER:
                    self.prf.toc()
                    self.prf.tic()
                ret_dict[CID_TO_TYPE[cid]].append(encrypted_sid)
        ret_dict = {t: pyobj2bytes(lst) for t, lst in ret_dict.items()}
        concatenated_parameters = pack_local_modules([[o.detach().numpy() for o in self.pp_lm_dict[t].parameters()]
                                                      for t in INDEX_TO_TYPE])
        if ENABLE_PROFILER:
            self.prf.toc(is_overhead=False)
            self.prf.upload([masked_wx, labels], [masked_wx, labels])
            org_ret_dict = {t: [] for t in INDEX_TO_TYPE}
            for cids, sample_id in ids:
                for cid in cids:
                    org_ret_dict[CID_TO_TYPE[cid]].append(sample_id.encode('ascii'))
            self.prf.upload(ret_dict, org_ret_dict)

        return [masked_wx, labels] + concatenated_parameters, 0, ret_dict

    def stage2(self, server_rnd, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        if ENABLE_PROFILER:
            self.prf.download(parameters[0], parameters[0], not_in_test=True)
            self.prf.download(server_rnd, server_rnd, not_in_test=True)
            self.prf.tic()
        self.recv_grad = torch.tensor(parameters[0])
        for party_type in self.type2pp_grads.keys():
            self.type2pp_grads[party_type] = np.zeros(self.pp_lm_sizes[party_type], dtype=int)
        if ENABLE_PROFILER:
            self.prf.toc(not_in_test=True)
            self.prf.tic()
        # update Active Party's local module
        self.intermediate_output.backward(self.recv_grad)
        self.ap_optimiser.step()
        self.ap_optimiser.zero_grad()
        if ENABLE_PROFILER:
            self.prf.toc(is_overhead=False, not_in_test=True)
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
            self.party_type = CID_TO_TYPE[cid]
            self.singleton_flag = len(PASSIVE_PARTY_CIDs[self.party_type]) == 1
            self.lm = generate_passive_party_local_module(self.party_type)
            self.optimiser = optim.SGD(self.lm.parameters(), lr=LEARNING_RATE, momentum=0.9)
            self.no_sample_selected = False
            self.output_shape = ()
            self.lm_size = sum([param.numel() for param in self.lm.parameters()])

    def setup_round1(self, parameters, config, t: Tuple[List[np.ndarray], int, dict]):
        if ENABLE_PROFILER:
            self.prf.tic()
        self.selector = get_sample_selector(self.cid)
        with torch.no_grad():
            _input = self.selector.select([0])
            self.output_shape = self.lm(_input).shape[1:]
        t[0].append(np.array([pyobj2bytes((self.cid, self.selector.get_range()))]))
        if ENABLE_PROFILER:
            self.prf.toc(is_overhead=False)
            self.prf.upload(t[0][-1], t[0][-1])
        if DEBUG:
            logger.info(f'client {self.cid}: upload bank list of size {len(t[0][0]) - 1}')

    def setup_round2(self, parameters, config, t: Tuple[List[np.ndarray], int, dict]):
        self.bwd_rng_dict = {cid: rng for cid, rng in self.bwd_rng_dict.items()
                             if cid in PASSIVE_PARTY_CIDs[self.party_type]}

    def stage1(self, server_rnd, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        logger.info(f"Client {self.cid}: reading encrypted batch and computing masked results...")
        if ENABLE_PROFILER:
            # If the client is the only one of its cluster,
            # no need to download parameters as they are updated locally
            if not self.singleton_flag:
                self.prf.download(parameters, parameters, not_in_test=True)
            self.prf.download(server_rnd, server_rnd, not_in_test=True)
            prv_config = config
            self.prf.tic()
        # For the same reason, the singleton client won't set parameters.
        if not self.singleton_flag:
            for param, given_param in zip(self.lm.parameters(), parameters):
                param.data = torch.tensor(given_param)
        # retrieve the real config of type List[bytes]
        config = bytes2pyobj(config[self.party_type])
        if ENABLE_PROFILER:
            self.prf.toc(is_overhead=False)
            self.prf.tic()
        # try read batch
        key = self.shared_secret_dict['0']
        self.mask = torch.zeros(len(config), dtype=bool)
        ids = []
        for i, encrypted_id in enumerate(config):
            # logger.info(f'try decrypting {obj_bytes} of type {type(obj_bytes)}')
            if ENABLE_PROFILER:
                self.prf.toc(is_overhead=False)
                self.prf.tic()
            sample_id = try_decrypt_and_load(key, encrypted_id)
            if ENABLE_PROFILER:
                self.prf.toc()
                self.prf.tic()
            if sample_id is not None:
                self.mask[i] = True
                ids += [(i, sample_id.decode('ascii'))]
        ids = sorted(ids, key=lambda x: x[0])
        if ENABLE_PROFILER:
            self.prf.toc(is_overhead=False)
            self.prf.download(prv_config, [sid for _, sid in ids])
            self.prf.tic()
        partial_batch_data = self.selector.select([o[1] for o in ids])
        self.no_sample_selected = True if len(ids) == 0 else False

        # expand the first dimension of the intermediate output to batch_size, padding zeros
        expanded_output = torch.zeros(len(config), *self.output_shape)
        if not self.no_sample_selected:
            self.intermediate_output = self.lm(partial_batch_data)
            expanded_output[self.mask] = self.intermediate_output
        if ENABLE_PROFILER:
            self.prf.toc(is_overhead=False)
            self.prf.tic()
        if VALIDATION:
            expanded_output = torch.zeros_like(expanded_output)
        expanded_output = quantize([expanded_output.detach().numpy()], CLIP_RANGE, TARGET_RANGE)[0]
        # masked_ret = masking(server_rnd, expanded_output, self.cid, self.shared_seed_dict)
        masked_ret = rng_masking(expanded_output, self.cid, self.fwd_rng_dict)
        if ENABLE_PROFILER:
            self.prf.toc()
            self.prf.upload(masked_ret, expanded_output)
        if DEBUG:
            logger.info(f'client {self.cid}: masking offset = {server_rnd}')
        return [masked_ret], 0, {}

    def stage2(self, server_rnd, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        if ENABLE_PROFILER:
            self.prf.download(server_rnd, server_rnd, not_in_test=True)
            self.prf.download(parameters[0], parameters[0], not_in_test=True)
            self.prf.tic()
        client_grad = torch.tensor(parameters[0])
        # bank client code
        if not self.no_sample_selected:
            self.intermediate_output.backward(client_grad[self.mask])
            # If the passive party only has this client, skip uploading grad and apply updates locally.
            if self.singleton_flag:
                self.optimiser.step()
                self.optimiser.zero_grad()
            else:
                self.partial_grad = []
                for param in self.lm.parameters():
                    self.partial_grad += [param.grad.cpu().detach().numpy().flatten()]
                self.partial_grad = np.concatenate(self.partial_grad)
                self.lm.zero_grad()
        else:
            self.partial_grad = np.zeros(self.lm_size)
        if VALIDATION:
            self.partial_grad = np.zeros_like(self.partial_grad)
        if ENABLE_PROFILER:
            self.prf.toc(is_overhead=False, not_in_test=True)
            self.prf.tic()
        if not self.singleton_flag:
            self.partial_grad = quantize([self.partial_grad], CLIP_RANGE, TARGET_RANGE)[0]
            masked_partial_grad = rng_masking(self.partial_grad, self.cid, self.bwd_rng_dict)
        if ENABLE_PROFILER:
            self.prf.toc(not_in_test=True)
            if not self.singleton_flag:
                self.prf.upload(masked_partial_grad, masked_partial_grad, not_in_test=True)
        if DEBUG:
            logger.info(f'client {self.cid}: uploading masked grad : {masked_partial_grad}')
        else:
            logger.info(f'client {self.cid}: uploading masked gradient')
        if ENABLE_PROFILER and server_rnd == 2 * TRAINING_ROUNDS - 1:
            self.total_prf.toc()
            txt = f'TRAIN:\ntotal_cpu_time = {self.total_prf.get_cpu_time()}, ' \
                  f'overhead = {self.prf.get_cpu_time() - self.prf.get_cpu_time(include_overhead=False)}\n' \
                  f'total_download_bytes = {self.prf.get_num_download_bytes()}, ' \
                  f'overhead = {self.prf.get_num_download_bytes() - self.prf.get_num_download_bytes(include_overhead=False)}\n' \
                  f'total_upload_bytes = {self.prf.get_num_upload_bytes()}, ' \
                  f'overhead = {self.prf.get_num_upload_bytes() - self.prf.get_num_upload_bytes(include_overhead=False)}\n' \
                  f'TEST:\ntotal_cpu_time = {self.prf.get_cpu_time(test_phase=True)}, ' \
                  f'overhead = {self.prf.get_cpu_time(test_phase=True) - self.prf.get_cpu_time(include_overhead=False, test_phase=True)}\n' \
                  f'total_download_bytes = {self.prf.get_num_download_bytes(test_phase=True)}, ' \
                  f'overhead = {self.prf.get_num_download_bytes(test_phase=True) - self.prf.get_num_download_bytes(include_overhead=False, test_phase=True)}\n' \
                  f'total_upload_bytes = {self.prf.get_num_upload_bytes(test_phase=True)}, ' \
                  f'overhead = {self.prf.get_num_upload_bytes(test_phase=True) - self.prf.get_num_upload_bytes(include_overhead=False, test_phase=True)}'

            logger.info(f'\n========\nclient {self.cid}:\n{txt}\n========\n')
        if self.singleton_flag:
            return [], 0, {}
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
