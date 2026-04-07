from game.board import Board
from game.board_utils import CARDS, get_5_random_cards, Move
import random
import time
from network.input import get_nn_training_data, get_nn_input
import torch
from network.model import OnitamaNet
import os
import yaml
from mcts.batched_mcts import BatchedMCTS
import torch.nn.functional as F
import gzip
import shutil
from tqdm import tqdm
import argparse
import json
import numpy as np
from tqdm import tqdm


def run_benchmark(args):
    config_path = os.path.join("benchmark", "configs", args.config)
    log_file = args.output

    config = yaml.safe_load(open(config_path, "r"))

    n_simulations = [n for n in config["n_simulations"]]
    n_runs = config["n_runs"]

    log = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = yaml.safe_load(open(os.path.join("models", "configs", "resnet.yaml"), "r"))

    model = OnitamaNet(config).to(device)

    mcts = BatchedMCTS(model, config, device)

    pbar = tqdm(desc="Running simulations VS time benchmark", total = len(n_simulations) * n_runs)
    for n in n_simulations:

        mcts.num_simulations = n
        results = np.empty((n_runs))

        for r in range(n_runs):
            pbar.update(1)

            boards = [Board(get_5_random_cards(cards = "all")) for _ in range(64)]

            start = time.time()

            _ = mcts.search_batch(boards)[0]

            total_time = time.time() - start

            results[r] = total_time

        mean = np.mean(results)
        std = np.std(results)

        log[str(n)] = {
            "mean" : mean,
            "std" : std
        }
    
    pbar.close()

    json.dump(log, open(log_file, "w"), indent=4)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Number of simulations VS time")
    parser.add_argument("--config", "-c", type = str, default = "simulation_cost.yaml", 
                        help = "path to the config file")
    parser.add_argument("--output", "-o", type = str, default = "benchmark/output_simulation_cost.json", 
                        help = "path of the output")
    
    args = parser.parse_args()

    run_benchmark(args)

