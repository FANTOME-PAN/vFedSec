import os
import shutil
import time
from pathlib import Path

import flwr as fl
from flwr.server import Server, ServerConfig, SimpleClientManager
from loguru import logger

from core.solution_federated import train_strategy_factory, train_client_factory, TRAINING_ROUNDS

parameters = {
    'num_clients_per_round': 5,
    'num_total_clients': 5,
    'num_rounds': TRAINING_ROUNDS,
    'data_dir': './client_data'
}


def bank_client_fn(cid):
    dir_pth = Path('data/bank')
    return train_client_factory(cid, dir_pth / f'p{cid}_data.csv', Path(f'client_data/party_{cid}'))


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
        "num_gpus": 0
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


if __name__ == '__main__':
    lt = time.localtime()
    # log_pth = f'logs/log[{lt.tm_hour}-{lt.tm_min}][{lt.tm_mday}-{lt.tm_mon}-{lt.tm_year}].log'
    logger.add(f'logs/server_log[{lt.tm_hour}-{lt.tm_min}][{lt.tm_mday}-{lt.tm_mon}-{lt.tm_year}].log',
               format="{time} {level} {message}", level="INFO")
    # # create folder for client data
    data_dir = Path(parameters['data_dir'])
    for filename in os.listdir(data_dir):
        party_dir_pth = data_dir / filename
        shutil.rmtree(party_dir_pth)

    run_demo()

    # # clean up caches
    for filename in os.listdir(data_dir):
        party_dir_pth = data_dir / filename
        shutil.rmtree(party_dir_pth)
