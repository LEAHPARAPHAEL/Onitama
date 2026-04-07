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

    batch_sizes = [batch_size for batch_size in config["batch_sizes"]]
    n_runs = config["n_runs"]

    log = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = yaml.safe_load(open(os.path.join("models", "configs", "resnet.yaml"), "r"))

    model = OnitamaNet(config).to(device)

    mcts = BatchedMCTS(model, config, device)

    mcts.num_simulations = 100

    pbar = tqdm(desc="Running batch_size VS time benchmark", total = len(batch_sizes) * n_runs)
    for batch_size in batch_sizes:

        results_batched = np.empty((n_runs))
        results_unbatched = np.empty((n_runs))

        for r in range(n_runs):
            pbar.update(1)

            boards = [Board(get_5_random_cards(cards = "all")) for _ in range(batch_size)]

            # Bench batched boards
            start = time.time()

            _ = mcts.search_batch(boards)[0]

            total_time = time.time() - start

            results_batched[r] = total_time

            for i in range(batch_size):
                mcts.reset_tree(i)


            # Bench unbatched boards
            start = time.time()

            for board in boards:
                _ = mcts.search_batch([board])[0]
                mcts.reset_tree(0)

            total_time = time.time() - start

            results_unbatched[r] = total_time

            for i in range(batch_size):
                mcts.reset_tree(i)



        mean_batched = np.mean(results_batched)
        std_batched = np.std(results_batched)

        mean_unbatched = np.mean(results_unbatched)
        std_unbatched = np.std(results_unbatched)

        log[str(batch_size)] = {
            "mean_batched" : mean_batched,
            "std_batched" : std_batched,
            "mean_unbatched" : mean_unbatched,
            "std_unbatched" : std_unbatched
        }
    
    pbar.close()

    json.dump(log, open(log_file, "w"), indent=4)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Number of simulations VS time")
    parser.add_argument("--config", "-c", type = str, default = "batch_time.yaml", 
                        help = "path to the config file")
    parser.add_argument("--output", "-o", type = str, default = "benchmark/output_batch_time.json", 
                        help = "path of the output")
    
    args = parser.parse_args()

    run_benchmark(args)