import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
import numpy as np
import pandas as pd
import torch
from flwr.common import FitIns, FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays, \
    EvaluateIns, EvaluateRes
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from loguru import logger
from torch import nn, optim

from core.client_template import TrainClientTemplate
from core.utils import pyobj2bytes, bytes2pyobj
from core.my_profiler import IProfiler, get_profiler
from core.secagg import quantize, reverse_quantize, encrypt, decrypt
from data_loaders import get_data_loader, get_sample_selector, IDataLoader, ISampleSelector
from models import generate_active_party_local_module, generate_passive_party_local_module, \
    generate_global_module, get_criterion
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
        self.agg_grad = np.zeros(26)
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
        config_dict = {"round": server_round}
        logger.info(f"[START] server round {server_round}")
        if DEBUG:
            logger.info(f'GPU STATUS: {torch.cuda.is_available()}')
        # rnd 1
        # collect public keys
        if server_round == 1:
            # add all cids to the config_dict as keys
            config_dict.update(dict(zip(cid_dict.keys(), [0] * len(cid_dict))))
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
                       , config=dict(list(self.fwd_dict[cid].items()) + list(config_dict.items())))
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
            fit_ins = FitIns(parameters=ndarrays_to_parameters([self.agg_grad]), config=config_dict)
            ins_lst = [(cid_dict['0'], fit_ins)]
        elif self.stage == 1:
            logger.info(f"stage 1: broadcasting model weights and encrypted batch to passive parties")
            # broadcast the weights and the encrypted batch to all bank clients
            config_dict.update(self.encrypted_batch)
            fit_ins = FitIns(parameters=ndarrays_to_parameters(self.weights), config=config_dict)
            ins_lst = [(proxy, fit_ins) for cid, proxy in cid_dict.items() if cid != '0']
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
            self.fwd_dict = dict([(o.cid, {}) for o, _ in results])
            for client, res in results:
                for other_cid, pk_bytes in res.metrics.items():
                    self.fwd_dict[other_cid][client.cid] = pk_bytes
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
                self.label = torch.from_numpy(self.label)
                self.intermediate_output = masked_wx
                # logger.info(f"logit type {type(self.logit)}, dtype {self.logit.dtype}")
                # broadcast to all bank clients
                self.encrypted_batch = results[0][1].metrics
                if DEBUG and LOGIC_TEST:
                    logger.info(f"server received encrypted batch:\n {self.encrypted_batch}")
            # if server_round == self.num_rnds, do nothing
            pass
        elif self.stage == 1:
            # receive masked results from all bank clients

            for proxy, res in results:
                masked_res = parameters_to_ndarrays(res.parameters)[0]
                self.intermediate_output += masked_res
                # logger.info(f"logit type {type(self.logit)}, dtype {self.logit.dtype}")
                # logger.info(f"masked_res type {type(masked_res)}, dtype {masked_res.dtype}")
            self.intermediate_output &= 0xffffffff
            self.intermediate_output = reverse_quantize([self.intermediate_output], CLIP_RANGE, TARGET_RANGE)[0]
            self.intermediate_output -= len(results) * CLIP_RANGE
            if DEBUG:
                logger.info(f'server: reconstructed output = {self.intermediate_output}')
                logger.info(f'server: labels = {self.label}')
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
            self.agg_grad = None
            for client, res in results:
                if client.cid == '0':
                    continue
                t = parameters_to_ndarrays(res.parameters)
                if DEBUG:
                    logger.info(f'server received {t}')
                if self.agg_grad is None:
                    self.agg_grad = t[0]
                else:
                    self.agg_grad += t[0]
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


# class TestSwiftClient(TrainClientTemplate):
#     """Custom Flower NumPyClient class for test."""
#
#     def __init__(
#             self, cid: str, df_pth: Path, client_dir: Path,
#             preds_format_path: Path,
#             preds_dest_path: Path,
#     ):
#         super().__init__(cid, df_pth, None, client_dir)
#         # UNIMPLEMENTED CODE HERE
#         self.weights = np.random.rand(28)
#         if LOGIC_TEST:
#             self.weights = np.zeros(28)
#         self.loader = TestDataLoader()
#         self.bank2cid: List[Tuple[Set[str], str]] = []
#         self.proba = np.array(0)
#         self.preds_format_path = preds_format_path
#         self.preds_dest_path = preds_dest_path
#         self.index = None
#
#     def setup_round2(self, parameters, config, t: Tuple[List[np.ndarray], int, dict]):
#         logger.info("Client 0: build bank to cid dict")
#         self.bank2cid = [(set(lst[1:]), lst[0]) for lst in parameters]
#         df = pd.read_csv(self.df_pth, index_col='MessageId')
#         self.index = df.index
#         self.loader.set(df, self.bank2cid)
#         logger.info("swift client: test XGBoost")
#         # training code
#         # UNIMPLEMENTED CODE HERE
#         if not LOGIC_TEST:
#             # self.loader.set_proba(np.zeros(len(df)))
#             all_proba = test_swift(df, self.client_dir)[1]
#             self.loader.set_proba(all_proba)
#         else:
#             self.loader.set_proba(np.zeros(len(df)))
#
#     def stage0(self, server_rnd, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
#         logger.info('swift client: preparing data...')
#         proba, batch = self.loader.get_data()
#         # if LOGIC_TEST:
#         #     proba = np.zeros(x.shape[0])
#         # else:
#         #     proba = predict_proba(self.net, x)
#         self.proba = proba
#         if DEBUG:
#             logger.info(f'swift predicted proba: {proba}')
#         # wx = w_0 * proba + b, where b = weights[27]
#         wx = quantize([self.weights[0] * proba + self.weights[-1]], CLIP_RANGE, TARGET_RANGE)[0]
#         masked_wx = masking(server_rnd + 1, wx, self.cid, self.shared_seed_dict)
#         if DEBUG:
#             logger.info(f'client {self.cid}: masking offset = {server_rnd + 1}')
#         ret_dict = {}
#         first_cid = None
#         for first_cid in self.shared_secret_dict:
#             break
#         for i, (sender_cid, receiver_cid, ordering_account, beneficiary_account) in enumerate(batch):
#             if sender_cid == 'missing':
#                 if DEBUG:
#                     logger.info(f'swift client: OrderingAccount {ordering_account} belongs to a missing bank.'
#                                 f' send to client {first_cid} instead')
#                 oa_seed = self.shared_secret_dict[first_cid]
#             else:
#                 oa_seed = self.shared_secret_dict[sender_cid]
#
#             if receiver_cid == 'missing':
#                 if DEBUG:
#                     logger.info(f'swift client: BeneficiaryAccount {beneficiary_account} belongs to a missing bank.'
#                                 f' send to client {first_cid} instead')
#                 ba_seed = self.shared_secret_dict[first_cid]
#             else:
#                 ba_seed = self.shared_secret_dict[receiver_cid]
#
#             cipher_oa = encrypt(oa_seed,
#                                 pyobj2bytes(ordering_account))
#             cipher_ba = encrypt(ba_seed,
#                                 pyobj2bytes(beneficiary_account))
#             t = (cipher_oa, cipher_ba)
#             ret_dict[str(i)] = pyobj2bytes(t)
#
#         # UNIMPLEMENTED CODE HERE
#         # labels = x[:, -1].astype(int)
#         return [masked_wx, self.weights[1:-1]], 0, ret_dict
#
#     def stage2(self, server_rnd, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
#         y_pred = parameters[0]
#         final_preds = pd.Series(y_pred, index=self.index)
#         preds_format_df = pd.read_csv(self.preds_format_path, index_col="MessageId")
#         preds_format_df["Score"] = preds_format_df.index.map(final_preds)
#         preds_format_df["Score"] = preds_format_df["Score"].astype(np.float64)
#         logger.info("Writing out test predictions...")
#         preds_format_df.to_csv(self.preds_dest_path)
#         logger.info("Done.")
#         return [], 0, {}
#
#
# class TestBankClient(TrainClientTemplate):
#     """Custom Flower NumPyClient class for training."""
#
#     def __init__(
#             self, cid: str, df_pth: Path, client_dir: Path
#     ):
#         super().__init__(cid, df_pth, None, client_dir)
#         self.weights = np.array(0)
#         self.account2flag = None
#
#     def _get_flag(self, account: str):
#         return self.account2flag.setdefault(account, 12)
#
#     def setup_round1(self, parameters, config, t: Tuple[List[np.ndarray], int, dict]):
#         df = pd.read_csv(self.df_pth, dtype=pd.StringDtype())
#         self.account2flag = dict(zip(df['Account'], df['Flags'].astype(int)))
#         t[0].append(np.array([self.cid] + list(df['Bank'].unique()), dtype=str))
#         if DEBUG:
#             logger.info(f'client {self.cid}: upload bank list of size {len(t[0][0]) - 1}')
#
#     def stage1(self, server_rnd, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
#         logger.info(f"Client {self.cid}: reading encrypted batch and computing masked results...")
#         # logger.info(f"client {self.cid}: received {str(config)}")
#         # the dim of weights is 26 at bank client. the weight of proba and the bias are kept in swift client
#         # weights[0] to [12] correspond to weights for ordering account flags
#         # weights[13] to [25] correspond to weights for beneficiary account flags
#         self.weights = parameters[0]
#         # try read batch
#         ret = np.zeros(len(config))
#         key = self.shared_secret_dict['swift']
#         # for ret[i] = optional(w2 * x2) + optional(w3 * x3)
#         # w2 = weights[:13], w3 = weights[13:]
#         # x2, x3 is the one-hot encoding of OA flag, BA flag
#         for str_i, obj_bytes in config.items():
#             # logger.info(f'try decrypting {obj_bytes} of type {type(obj_bytes)}')
#             cipher_oa, cipher_ba = bytes2pyobj(obj_bytes)
#             i = int(str_i)
#             if (oa := try_decrypt_and_load(key, cipher_oa)) is not None:
#                 flg = self._get_flag(oa)
#                 ret[i] += self.weights[flg]
#                 if DEBUG and LOGIC_TEST:
#                     logger.info(f'BATCH_IDX: {i} ORDERING_ACCOUNT')
#             if (ba := try_decrypt_and_load(key, cipher_ba)) is not None:
#                 flg = self._get_flag(ba)
#                 ret[i] += self.weights[flg + 13]
#                 if DEBUG and LOGIC_TEST:
#                     logger.info(f'BATCH_IDX: {i} BENEFICIARY_ACCOUNT')
#         ret = quantize([ret], CLIP_RANGE, TARGET_RANGE)[0]
#         masked_ret = masking(server_rnd, ret, self.cid, self.shared_seed_dict)
#         if DEBUG:
#             logger.info(f'client {self.cid}: masking offset = {server_rnd}')
#         return [masked_ret], 0, {}
#
#     def stage2(self, server_rnd, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
#         return [], 0, {}
#
#
# def test_client_factory(
#         cid: str,
#         data_path: Path,
#         client_dir: Path,
#         preds_format_path: Path,
#         preds_dest_path: Path,
# ) -> Union[fl.client.Client, fl.client.NumPyClient]:
#     """
#     Factory function that instantiates and returns a Flower Client for test-time
#     inference. The federated learning simulation engine will use this function
#     to instantiate clients with all necessary dependencies.
#
#     Args:
#         cid (str): Identifier for a client node/federation unit. Will be
#             constant over the simulation and between train and test stages. The
#             SWIFT node will always be named 'swift'.
#         data_path (Path): Path to CSV test data file specific to this client.
#         client_dir (Path): Path to a directory specific to this client that is
#             available over the simulation. Clients can use this directory for
#             saving and reloading client state.
#         preds_format_path (Optional[Path]): Path to CSV file matching the format
#             you must write your predictions with, filled with dummy values. This
#             will only be non-None for the 'swift' client—bank clients should not
#             write any predictions and receive None for this argument.
#         preds_dest_path (Optional[Path]): Destination path that you must write
#             your test predictions to as a CSV file. This will only be non-None
#             for the 'swift' client—bank clients should not write any predictions
#             and will receive None for this argument.
#
#     Returns:
#         (Union[Client, NumPyClient]): Instance of Flower Client or NumPyClient.
#     """
#     if cid == "swift":
#         logger.info("Initializing SWIFT client for {}", cid)
#         return TestSwiftClient(
#             cid,
#             df_pth=data_path,
#             client_dir=client_dir,
#             preds_format_path=preds_format_path,
#             preds_dest_path=preds_dest_path,
#         )
#     else:
#         logger.info("Initializing bank client for {}", cid)
#         return TestBankClient(cid, df_pth=data_path, client_dir=client_dir)
#
#
# class TestStrategy(fl.server.strategy.Strategy):
#     """Custom Flower strategy for test."""
#
#     def __init__(self, server_dir: Path):
#         self.server_dir = server_dir
#         self.public_keys_dict = {}
#         self.fwd_dict = {}
#         self.agg_grad = np.zeros(26)
#         self.stage = 0
#         self.label = np.array(0)
#         self.weights = np.array(0)
#         self.logit = np.array(0)
#         self.preds = np.array(0)
#         self.encrypted_batch: Dict[str, bytes] = {}
#         self.cached_banklsts = []
#         self.rnd_cnt = 0
#         super().__init__()
#
#     def initialize_parameters(self, client_manager: ClientManager) -> Parameters:
#         """Do nothing. Return empty Flower Parameters dataclass."""
#         return empty_parameters()
#
#     def __configure_round(
#             self,
#             server_round: int,
#             parameters: Parameters,
#             client_manager: ClientManager
#     ) -> List[Tuple[ClientProxy, FitIns]]:
#         """Configure the next round of training."""
#         cid_dict: Dict[str, ClientProxy] = client_manager.all()
#         config_dict = {"round": server_round}
#         logger.info(f"[START] server round {server_round}")
#         if DEBUG:
#             logger.info(f'GPU STATUS: {torch.cuda.is_available()}')
#         # rnd 1
#         # collect public keys
#         if server_round == 1:
#             # add all cids to the config_dict as keys
#             config_dict = dict(zip(cid_dict.keys(), [0] * len(cid_dict))) | config_dict
#             logger.info(f"server's requesting public keys...")
#             if DEBUG:
#                 logger.info(f"send to clients {str(config_dict)}")
#             fit_ins = FitIns(parameters=empty_parameters(), config=config_dict)
#             return [(o, fit_ins) for o in cid_dict.values()]
#         # rnd 2
#         # broadcast public keys, swift train
#         if server_round == 2:
#             # forward public keys to corresponding clients
#             logger.info(f"server's forwarding public keys...")
#             if DEBUG:
#                 for cid in self.fwd_dict:
#                     logger.info(f'forward to {cid} {str(self.fwd_dict[cid].keys())}')
#             ins_lst = [(
#                 proxy,
#                 FitIns(parameters=empty_parameters() if proxy.cid != 'swift' else self.cached_banklsts
#                        , config=self.fwd_dict[cid] | config_dict)
#             ) for cid, proxy in cid_dict.items()]
#             if DEBUG:
#                 logger.info(f"server's sending to swift bank lists {str(parameters_to_ndarrays(self.cached_banklsts))}")
#             self.cached_banklsts = []
#             self.fwd_dict = {}
#             return ins_lst
#         # rnd 3 -> N
#         # joint train
#         config_dict['stage'] = self.stage
#         if self.stage == 0:
#             fit_ins = FitIns(parameters=ndarrays_to_parameters([self.agg_grad]), config=config_dict)
#             ins_lst = [(cid_dict['swift'], fit_ins)]
#         elif self.stage == 1:
#             logger.info(f"stage 1: broadcasting model weights and encrypted batch to bank clients")
#             # broadcast the weights and the encrypted batch to all bank clients
#             config_dict = config_dict | self.encrypted_batch
#             fit_ins = FitIns(parameters=ndarrays_to_parameters([self.weights]), config=config_dict)
#             ins_lst = [(proxy, fit_ins) for cid, proxy in cid_dict.items() if cid != 'swift']
#         elif self.stage == 2:
#             logger.info(f"stage 2: send y_pred to swift")
#             fit_ins = FitIns(parameters=ndarrays_to_parameters([self.preds]), config=config_dict)
#             ins_lst = [(cid_dict['swift'], fit_ins)]
#         else:
#             raise AssertionError("Stage number should be 0, 1, or 2")
#
#         return ins_lst
#
#     def __aggregate_round(
#             self,
#             server_round: int,
#             results: List[Tuple[ClientProxy, FitRes]],
#             failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
#     ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
#         """Aggregate training results."""
#         if (n_failures := len(failures)) > 0:
#             logger.error(f"Had {n_failures} failures in round {server_round}")
#             raise Exception(f"Had {n_failures} failures in round {server_round}")
#         # rnd 1
#         # gather all public keys, and forward them to corresponding client
#         if server_round == 1:
#             logger.info(f'server\'s processing received bank lists')
#             # bank client would store the list of bank names in the parameters field
#             self.cached_banklsts = [parameters_to_ndarrays(res.parameters)[0]
#                                     for proxy, res in results if proxy.cid != 'swift']
#             self.cached_banklsts = ndarrays_to_parameters(self.cached_banklsts)
#             # fwd_dict[to_client][from_client] = public key to to_client generated by from_client
#             logger.info(f'server\'s creating forward dict')
#             self.fwd_dict = dict([(o.cid, {}) for o, _ in results])
#             for client, res in results:
#                 for other_cid, pk_bytes in res.metrics.items():
#                     self.fwd_dict[other_cid][client.cid] = pk_bytes
#             logger.info(f"[END] server round {server_round}")
#             return None, {}
#         # rnd 2
#         # do nothing
#         if server_round == 2:
#             logger.info(f"[END] server round {server_round}")
#             return None, {}
#         # rnd 3 -> N, joint training
#         if self.stage == 0:
#             masked_wx, weights = parameters_to_ndarrays(results[0][1].parameters)
#             self.logit = masked_wx
#             self.weights = weights
#             # broadcast to all bank clients
#             self.encrypted_batch = results[0][1].metrics
#             logger.info(f"server received encrypted batch")
#             # if server_round == self.num_rnds, do nothing
#             pass
#         elif self.stage == 1:
#             # receive masked results from all bank clients
#             for proxy, res in results:
#                 masked_res = parameters_to_ndarrays(res.parameters)[0]
#                 self.logit += masked_res
#             self.logit &= 0xffffffff
#             self.logit = reverse_quantize([self.logit], CLIP_RANGE, TARGET_RANGE)[0]
#             self.logit -= len(results) * CLIP_RANGE
#             if DEBUG:
#                 logger.info(f'server: reconstructed logits = {self.logit}')
#
#             tmp = np.exp(-self.logit)
#             y_pred = 1. / (1. + tmp)
#             self.preds = y_pred
#             pass
#         elif self.stage == 2:
#             pass
#         else:
#             raise AssertionError("Stage number should be 0, 1, or 2")
#         self.stage = (self.stage + 1) % 3
#         return None, {}
#
#     def configure_fit(
#             self,
#             server_round: int,
#             parameters: Parameters,
#             client_manager: ClientManager
#     ) -> List[Tuple[ClientProxy, FitIns]]:
#         """Configure the next round of training."""
#         self.rnd_cnt += 1
#         return self.__configure_round(self.rnd_cnt, parameters, client_manager)
#
#     def aggregate_fit(
#             self,
#             server_round: int,
#             results: List[Tuple[ClientProxy, FitRes]],
#             failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
#     ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
#         """Aggregate training results."""
#         return self.__aggregate_round(self.rnd_cnt, results, failures)
#
#     def configure_evaluate(
#             self,
#             server_round: int,
#             parameters: Parameters,
#             client_manager: ClientManager
#     ) -> List[Tuple[ClientProxy, EvaluateIns]]:
#         if server_round == 3:
#             logger.info('Test Strategy: skip last eval round!')
#             return []
#         self.rnd_cnt += 1
#         ins_lst = self.__configure_round(self.rnd_cnt, parameters, client_manager)
#         return [(proxy, EvaluateIns(ins.parameters, ins.config)) for proxy, ins in ins_lst]
#
#     def aggregate_evaluate(
#             self,
#             server_round: int,
#             results: List[Tuple[ClientProxy, EvaluateRes]],
#             failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
#     ) -> Tuple[Optional[float], Dict[str, Scalar]]:
#         if server_round == 3:
#             logger.info('Test Strategy: skip last eval round!')
#             return None, {}
#
#         def convert(res: EvaluateRes) -> FitRes:
#             params = bytes2pyobj(res.metrics.pop('parameters'))
#             return FitRes(res.status, params, res.num_examples, res.metrics)
#
#         results = [(proxy, convert(eval_res)) for proxy, eval_res in results]
#         return self.__aggregate_round(self.rnd_cnt, results, failures)
#
#     def evaluate(self, server_round, parameters):
#         """Not running any centralized evaluation."""
#         return None
#
#
# def test_strategy_factory(
#         server_dir: Path,
# ) -> Tuple[fl.server.strategy.Strategy, int]:
#     """
#     Factory function that instantiates and returns a Flower Strategy, plus the
#     number of federation rounds to run.
#
#     Args:
#         server_dir (Path): Path to a directory specific to the server/aggregator
#             that is available over the simulation. The server can use this
#             directory for saving and reloading server state. Using this
#             directory is required for the trained model to be persisted between
#             training and test stages.
#
#     Returns:
#         (Strategy): Instance of Flower Strategy.
#         (int): Number of federated learning rounds to execute.
#     """
#     test_strategy = TestStrategy(server_dir=server_dir)
#     # setup rounds = 2 / 2 = 1
#     # predict rounds = 3 / 2 = 1.5 (stage 0, 1, 2)
#     # num_rounds = 1 + 1.5 = 2.5
#     num_rounds = 3
#     return test_strategy, num_rounds
