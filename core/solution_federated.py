
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import time

import flwr as fl
import numpy as np
import torch
from flwr.common import FitIns, FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays, \
    EvaluateIns, EvaluateRes
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from loguru import logger
from torch import optim

from core.secagg import reverse_quantize
from core.utils import bytes2pyobj, unpack_local_modules
from models import generate_global_module, get_criterion
from settings import *

"""=== Utility Functions ==="""


def empty_parameters() -> Parameters:
    """Utility function that generates empty Flower Parameters dataclass instance."""
    return ndarrays_to_parameters([])


"""=== Strategy Class ==="""


class TrainStrategy(fl.server.strategy.Strategy):
    """Custom Flower strategy for training."""

    def __init__(self, server_dir: Path, num_rounds):
        self.server_dir = server_dir
        self.public_keys_dict = {}
        self.fwd_dict = {}
        self.agg_grad: List[np.ndarray] = []
        self.stage = 0
        self.num_rnds = TRAINING_ROUNDS << 1
        self.label = np.array(0)
        self.weights = np.array(0)
        self.intermediate_output = np.array(0)
        self.gm = generate_global_module()
        self.gm_optimizer = optim.SGD(self.gm.parameters(), lr=LEARNING_RATE, momentum=0.9)
        self.client_grad = np.array(0)
        self.encrypted_batch: Dict[str, bytes] = {}
        self.criterion = get_criterion()
        self.cached_ranges = []
        self.rnd_cnt = 0
        self.iter_cnt = 0
        super().__init__()

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters:
        """Do nothing. Return empty Flower Parameters dataclass."""
        return empty_parameters()

    def __configure_round(
            self,
            server_round: int,
            parameters: Parameters,
            client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, Union[FitIns]]]:
        """Configure the next round of training."""
        cid_dict: Dict[str, ClientProxy] = client_manager.all()
        config_dict: Dict[str, Scalar] = {"round": server_round}
        logger.info(f"[START] server round {server_round}")
        if DEBUG:
            logger.info(f'GPU STATUS: {torch.cuda.is_available()}')
        # rnd 1
        # collect public keys
        if server_round == 1:
            # add all cids to the config_dict as keys
            # config_dict.update(dict(zip(cid_dict.keys(), [0] * len(cid_dict))))
            lt = time.localtime()
            config_dict['log_path'] = f'logs/clients_log[{lt.tm_hour} {lt.tm_min} {lt.tm_sec}][{lt.tm_mday}-{lt.tm_mon}-{lt.tm_year}].log'
            logger.info(f"server's requesting public keys...")
            if DEBUG:
                logger.info(f"send to clients {str(config_dict)}")
            fit_ins = FitIns(parameters=empty_parameters(), config=config_dict)
            return [(o, fit_ins) for o in cid_dict.values()]
        # rnd 2
        # broadcast public keys, swift train
        if server_round == 2:
            # forward public keys to corresponding clients
            logger.info(f"server's forwarding public keys...")

            if DEBUG:
                for cid in self.fwd_dict:
                    logger.info(f'forward to {cid} {str(self.fwd_dict[cid].keys())}')
            ins_lst = [(
                proxy,
                FitIns(parameters=empty_parameters() if proxy.cid != '0' else self.cached_ranges
                       , config={**self.fwd_dict, **config_dict})
            ) for cid, proxy in cid_dict.items()]
            self.cached_ranges = []
            self.fwd_dict = {}
            return ins_lst
        # rnd 3 -> N
        # joint train
        config_dict['stage'] = self.stage
        if self.stage == 0:
            logger.info(f"stage 0: sending gradients to Client 0")
            if server_round == self.num_rnds:
                config_dict['stop'] = 1
                torch.save(self.gm, GLOBAL_MODULE_SAVE_PATH)
            fit_ins = FitIns(parameters=ndarrays_to_parameters(self.agg_grad), config=config_dict)
            ins_lst = [(cid_dict['0'], fit_ins)]
        elif self.stage == 1:
            logger.info(f"stage 1: broadcasting model weights and encrypted batch to passive parties")
            # broadcast the weights and the encrypted batch to all bank clients
            ins_lst = []
            # broadcast the weights and the encrypted batch to clients by type
            for idx, t in enumerate(INDEX_TO_TYPE):
                cfg = config_dict.copy()
                cfg[t] = self.encrypted_batch[t]
                fit_ins = FitIns(parameters=ndarrays_to_parameters(self.weights[idx]), config=cfg)
                ins_lst += [(proxy, fit_ins) for cid, proxy in cid_dict.items() if cid in PASSIVE_PARTY_CIDs[t]]

        elif self.stage == 2:
            if DEBUG:
                logger.info(f"stage 2: broadcasting client_grad to all clients {self.client_grad}")
            else:
                logger.info(f"stage 2: broadcasting beta to all clients")
            fit_ins = FitIns(parameters=ndarrays_to_parameters([self.client_grad]), config=config_dict)
            ins_lst = [(proxy, fit_ins) for cid, proxy in cid_dict.items()]
        else:
            raise AssertionError("Stage number should be 0, 1, or 2")

        return ins_lst

    def __aggregate_round(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if (n_failures := len(failures)) > 0:
            logger.error(f"Had {n_failures} failures in round {server_round}")
            raise Exception(f"Had {n_failures} failures in round {server_round}")
        # rnd 1
        # gather all public keys, and forward them to corresponding client
        if server_round == 1:
            logger.info(f'server\'s processing received bank lists')
            # bank client would store the list of bank names in the parameters field
            self.cached_ranges = [parameters_to_ndarrays(res.parameters)[0]
                                  for proxy, res in results if proxy.cid != '0']
            self.cached_ranges = ndarrays_to_parameters([np.concatenate(self.cached_ranges)])
            # fwd_dict[to_client][from_client] = public key to to_client generated by from_client
            logger.info(f'server\'s creating forward dict')
            self.fwd_dict = {}
            for client, res in results:
                self.fwd_dict[client.cid] = res.metrics['pk']
                # for other_cid, pk_bytes in res.metrics.items():
                #     self.fwd_dict[other_cid][client.cid] = pk_bytes
            logger.info(f"[END] server round {server_round}")
            return None, {}
        # rnd 2
        # do nothing
        if server_round == 2:
            logger.info(f"[END] server round {server_round}")
            return None, {}
        # rnd 3 -> N, joint training
        if self.stage == 0:
            if server_round < self.num_rnds:
                arrs = parameters_to_ndarrays(results[0][1].parameters)
                masked_wx, self.label, self.weights = arrs[0], arrs[1], arrs[2:]
                self.weights = unpack_local_modules(self.weights)
                self.label = torch.from_numpy(self.label)
                self.intermediate_output = masked_wx
                # logger.info(f"logit type {type(self.logit)}, dtype {self.logit.dtype}")
                # broadcast to all bank clients
                self.encrypted_batch = results[0][1].metrics
                if DEBUG:
                    logger.info(f"server received encrypted batch:\n {self.encrypted_batch}")
            # if server_round == self.num_rnds, do nothing
            pass
        elif self.stage == 1:
            # receive masked results from all bank clients

            for proxy, res in results:
                masked_res = parameters_to_ndarrays(res.parameters)[0]
                type_idx = TYPE_TO_INDEX[CID_TO_TYPE[proxy.cid]]
                start, end = type_idx * EMBEDDING_DIM, (type_idx + 1) * EMBEDDING_DIM
                self.intermediate_output[:, start: end] += masked_res
                # logger.info(f"logit type {type(self.logit)}, dtype {self.logit.dtype}")
                # logger.info(f"masked_res type {type(masked_res)}, dtype {masked_res.dtype}")
            self.intermediate_output &= 0xffffffff
            self.intermediate_output = reverse_quantize([self.intermediate_output], CLIP_RANGE, TARGET_RANGE)[0]
            self.intermediate_output -= len(results) * CLIP_RANGE
            if DEBUG:
                logger.info(f'server: reconstructed output = {self.intermediate_output}')
                logger.info(f'server: labels = {self.label}')
            if VALIDATION:
                logger.info(f"VALIDATION:\n{self.intermediate_output}")
            self.iter_cnt += 1
            self.intermediate_output = torch.tensor(self.intermediate_output, dtype=torch.float, requires_grad=True)
            logits = self.gm(self.intermediate_output)
            loss = self.criterion(logits, self.label)
            logger.info(f'Iteration {self.iter_cnt}: training loss = {loss.item()}')
            # compute gradients
            loss.backward()
            self.client_grad = self.intermediate_output.grad.clone().detach().numpy()
            self.gm_optimizer.step()
            self.gm_optimizer.zero_grad()
        elif self.stage == 2:
            self.agg_grad = [0 for _ in INDEX_TO_TYPE]
            for client, res in results:
                if client.cid == '0':
                    continue
                masked_grad = parameters_to_ndarrays(res.parameters)[0]
                idx = TYPE_TO_INDEX[CID_TO_TYPE[client.cid]]
                if DEBUG:
                    logger.info(f'server received {masked_grad}')
                self.agg_grad[idx] += masked_grad
                # if self.agg_grad is None:
                #     self.agg_grad = t[0]
                # else:
                #     self.agg_grad += t[0]
            pass
        else:
            raise AssertionError("Stage number should be 0, 1, or 2")
        self.stage = (self.stage + 1) % 3
        return None, {}

    def configure_fit(
            self,
            server_round: int,
            parameters: Parameters,
            client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        self.rnd_cnt += 1
        return self.__configure_round(self.rnd_cnt, parameters, client_manager)

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate training results."""
        return self.__aggregate_round(self.rnd_cnt, results, failures)

    def configure_evaluate(
            self,
            server_round: int,
            parameters: Parameters,
            client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        self.rnd_cnt += 1
        ins_lst = self.__configure_round(self.rnd_cnt, parameters, client_manager)
        return [(proxy, EvaluateIns(ins.parameters, ins.config)) for proxy, ins in ins_lst]

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:

        def convert(res: EvaluateRes) -> FitRes:
            params = bytes2pyobj(res.metrics.pop('parameters'))
            return FitRes(res.status, params, res.num_examples, res.metrics)

        results = [(proxy, convert(eval_res)) for proxy, eval_res in results]
        return self.__aggregate_round(self.rnd_cnt, results, failures)

    def evaluate(self, server_round, parameters):
        """Not running any centralized evaluation."""
        return None


def train_strategy_factory(
        server_dir: Path,
) -> Tuple[fl.server.strategy.Strategy, int]:
    """
    Factory function that instantiates and returns a Flower Strategy, plus the
    number of federated learning rounds to run.

    Args:
        server_dir (Path): Path to a directory specific to the server/aggregator
            that is available over the simulation. The server can use this
            directory for saving and reloading server state. Using this
            directory is required for the trained model to be persisted between
            training and test stages.

    Returns:
        (Strategy): Instance of Flower Strategy.
        (int): Number of federated learning rounds to execute.
    """
    # num_rounds of setup phase = 2 / 2 = 1
    # num_rounds of one training round = 3 / 2 = 1.5 (stage 0,1,2)
    # end round, end at stage 0, i.e., 0.5 round
    # num_rounds = 1 + 1.5N + 0.5 = 1.5(N+1)
    num_rounds = TRAINING_ROUNDS
    training_strategy = TrainStrategy(server_dir=server_dir, num_rounds=num_rounds)
    return training_strategy, num_rounds
