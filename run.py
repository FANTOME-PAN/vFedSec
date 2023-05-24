import os
import shutil
import time
from pathlib import Path

import flwr as fl
from flwr.server import Server, ServerConfig, SimpleClientManager
from loguru import logger

from core.solution_federated import train_strategy_factory
from core.client import train_client_factory
from settings import TRAINING_ROUNDS, CID_TO_DATA_PATH

parameters = {
    'num_clients_per_round': 5,
    'num_total_clients': 5,
    'num_rounds': TRAINING_ROUNDS,
    'data_dir': './client_data'
}


def bank_client_fn(cid):
    return train_client_factory(cid, CID_TO_DATA_PATH[cid], Path(f'client_data/party_{cid}'))


def start_simulation(
    client_fn,
    num_clients,
    client_resources,
    server,
    config,
    strategy,
):
    """Wrapper to always seed client selection."""
    return fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        client_resources=client_resources,
        server=server,
        config=config,
        strategy=strategy,
    )


def get_device() -> str:
    device = "cpu"
    # if torch.cuda.is_available():
    #     device = "cuda"
    return device


def run_demo():
    client_resources = {
        "num_gpus": 1
    }

    strategy = train_strategy_factory(Path('./'))[0]
    client_manager = SimpleClientManager()
    server = Server(
        client_manager=client_manager,
        strategy=strategy,
    )
    return start_simulation(
        client_fn=bank_client_fn,
        num_clients=parameters["num_total_clients"],
        client_resources=client_resources,
        server=server,
        config=ServerConfig(num_rounds=parameters["num_rounds"]),
        strategy=strategy
    )


def clear():
    # clean up caches
    data_dir = Path(parameters['data_dir'])
    for filename in os.listdir(data_dir):
        if filename == '.gitkeep':
            continue
        party_dir_pth = data_dir / filename
        shutil.rmtree(party_dir_pth)


if __name__ == '__main__':
    lt = time.localtime()
    logger.add(f'logs/server_log[{lt.tm_hour} {lt.tm_min} {lt.tm_sec}][{lt.tm_mday}-{lt.tm_mon}-{lt.tm_year}].log',
               format="{time} {level} {message}", level="INFO")
    clear()
    run_demo()
    clear()
