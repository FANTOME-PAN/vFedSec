import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
from flwr.common import Scalar
from loguru import logger

from core.utils import ndarrays_to_param_bytes
from core.my_profiler import get_profiler, IProfiler
from core.secagg import public_key_to_bytes, bytes_to_public_key, generate_key_pairs, generate_shared_key, \
    private_key_to_bytes, bytes_to_private_key
from settings import DEBUG, ENABLE_PROFILER


def sum_func(x):
    x += 1
    return sum(x)


class TrainClientTemplate(fl.client.NumPyClient):
    """Custom Flower NumPyClient class for training."""

    def __init__(
            self, cid: str, df_pth: Path, client_dir: Path
    ):
        super().__init__()
        self.cache_pth = client_dir / 'cache.pkl'
        self.cid = cid
        self.df_pth = df_pth
        self.client_dir = client_dir
        self.initialised = False
        if self.reload():
            self.initialised = True
        else:
            self.prf: IProfiler = get_profiler()
            self.total_prf = get_profiler()
            self.shared_secret_dict = {}
            self.shared_seed_dict = {}
            self.secret_key_dict = {}
            self.rnd_cnt = 0
            self.stage = -1
            self.partial_grad = np.array(0)
            self.recv_grad = np.array(0)
            self.intermediate_output = np.array(0)
            self.log_path = None

    def check_stage(self, stage):
        # init
        if self.stage == -1:
            if self.cid == '0':
                assert stage == 0
            else:
                assert stage == 1
        elif self.stage == 0:
            assert self.cid == '0'
            # swift client is not in stage 1
            assert stage == 2
        elif self.stage == 1:
            assert self.cid != '0'
            assert stage == 2
        elif self.stage == 2:
            if self.cid == '0':
                assert stage == 0
            else:
                # bank client is not in stage 0
                assert stage == 1
        self.stage = stage

    def setup(self, server_round, parameters, config):
        rnd = server_round
        # rnd 1
        # generate keys and reply with public keys
        if rnd == 1:
            self.log_path = config.pop('log_path')
            logger.add(self.log_path, format="{time} {level} {message}", level="INFO")
            logger.info(f'client {self.cid}: generating key pairs')
            if ENABLE_PROFILER:
                self.prf.tic()
            for cid in config:
                if cid == self.cid:
                    continue
                sk, pk = generate_key_pairs()
                self.secret_key_dict[cid] = private_key_to_bytes(sk)
                config[cid] = public_key_to_bytes(pk)
            config.pop(self.cid)
            t = ([], 0, config)
            if ENABLE_PROFILER:
                self.prf.toc()
                self.prf.upload(config)
            if DEBUG:
                logger.info(f'client {self.cid}: replying {str(config.keys())}')
            else:
                logger.info(f'client {self.cid}: uploading public keys, sized {sys.getsizeof(config)}')
            self.setup_round1(parameters, config, t)
            return t
        # rnd 2
        # generate shared secrets
        if rnd == 2:
            if DEBUG:
                logger.info(f'client {self.cid}: receiving public keys from {str(config.keys())}')
            else:
                logger.info(f'client {self.cid}: receiving public keys, sized {sys.getsizeof(config)}')
            # logger.info(f'client {self.cid}: keys of secret key dict {str(self.secret_key_dict.keys())}')
            if ENABLE_PROFILER:
                self.prf.download(config)
                self.prf.tic()
            for cid, pk_bytes in config.items():
                sk = bytes_to_private_key(self.secret_key_dict[cid])
                pk = bytes_to_public_key(pk_bytes)
                shared_key = generate_shared_key(sk, pk)
                self.shared_secret_dict[cid] = shared_key
                seed32 = 0
                for i in range(0, len(shared_key), 4):
                    seed32 ^= int.from_bytes(shared_key[i:i + 4], 'little')
                self.shared_seed_dict[cid] = np.array(seed32, dtype=np.int32)
            if ENABLE_PROFILER:
                self.prf.toc()
            if DEBUG:
                logger.info(f'client {self.cid}: shared seed {str(self.shared_seed_dict)}')
            t = ([], 0, {})
            self.setup_round2(parameters, config, t)
            self.secret_key_dict = {}
            return t

    def setup_round1(self, parameters, config, t: Tuple[List[np.ndarray], int, dict]):
        pass

    def setup_round2(self, parameters, config, t: Tuple[List[np.ndarray], int, dict]):
        pass

    def stage0(self, server_rnd, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        return [], 0, {}

    def stage1(self, server_rnd, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        return [], 0, {}

    def stage2(self, server_rnd, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        return [], 0, {}

    def get_vars(self):
        return self.__dict__

    def cache(self):
        # for k, v in vars(self).items():
        #     logger.info(f"client {self.cid}: saving {k}")
        #     torch.save(v, self.cache_pth)
        if not self.client_dir.exists():
            os.makedirs(self.client_dir)
        with open(self.cache_pth, 'wb') as f:
            pickle.dump(self.get_vars(), f)
        # torch.save(vars(self), self.cache_pth)

    def reload(self):
        if self.cache_pth.exists():
            logger.info(f'client {self.cid}: reloading from {str(self.cache_pth)}')
            with open(self.cache_pth, 'rb') as f:
                self.__dict__.update(pickle.load(f))
            return True
            # self.__dict__.update(torch.load(self.cache_pth))
        return False

    def __execute(self, parameters: List[np.ndarray], config: Dict[str, Scalar]) \
            -> Tuple[List[np.ndarray], int, dict]:
        rnd = config.pop('round')
        if rnd > self.rnd_cnt:
            self.rnd_cnt = rnd
        else:
            logger.error(f"Round error at Client {self.cid}: round {rnd} should have been completed.")
            raise RuntimeError(f"Round error: round {rnd} should have been completed.")

        if rnd <= 2:
            ret = self.setup(rnd, parameters, config)
            return ret

        # rnd 3 -> N, joint training
        self.check_stage(config.pop('stage'))
        # stage 0
        # SWIFT ONLY, reply with batch selection, w1 * proba, weights
        if self.stage == 0:
            ret = self.stage0(rnd, parameters, config)
            if 'stop' in config:
                logger.info('Client 0: stop signal is detected. abort federated training')

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

                ret = ([], 0, {})
        # stage 1
        # BANK ONLY, reply with masked wx
        elif self.stage == 1:
            ret = self.stage1(rnd, parameters, config)
        # stage 2
        elif self.stage == 2:
            ret = self.stage2(rnd, parameters, config)
        else:
            raise AssertionError()
        return ret

    def fit(
            self, parameters: List[np.ndarray], config: dict
    ) -> Tuple[List[np.ndarray], int, dict]:
        self.total_prf.tic()
        ret = self.__execute(parameters, config)
        self.total_prf.toc()
        self.cache()
        return ret

    def evaluate(
            self, parameters: List[np.ndarray], config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        self.total_prf.tic()
        params, num, metrics = self.__execute(parameters, config)
        metrics['parameters'] = ndarrays_to_param_bytes(params)
        self.total_prf.toc()
        self.cache()
        return 0., num, metrics
