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

    n_games_list = [n_games for n_games in config["n_games"]]
    n_runs = config["n_runs"]

    log = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = yaml.safe_load(open(os.path.join("models", "configs", "resnet.yaml"), "r"))

    model = OnitamaNet(config).to(device)

    mcts = BatchedMCTS(model, config, device)

    with gzip.open("models/weights/resnet/v100_625.pt.gz", "rb") as f:
        save_dict = torch.load(f, weights_only = False)
        
    model_state_dict = save_dict["model_state_dict"]
    model.load_state_dict(model_state_dict)
    model.eval()


    mcts.num_simulations = 100

    pbar = tqdm(desc="Running time VS number of games benchmark", total = len(n_games_list) * n_runs)
    for n_games in n_games_list:

        results_reuse = np.empty((n_runs))
        results_no_reuse = np.empty((n_runs))

        for r in range(n_runs):
            pbar.update(1)

            # Bench batched boards
            game_over_count = 0
            boards = [Board(get_5_random_cards(cards = "standard")) for _ in range(64)]
            for i in range(64):
                mcts.reset_tree(i)
            start = time.time()
            while game_over_count < n_games:
                policies = mcts.search_batch(boards)
                
                for i in range(64):
                    board = boards[i]
                    policy = policies[i]
                    
                    policy = F.relu(policy)
                    action_idx = torch.multinomial(policy, 1).item()
                    move = board.action_index_to_move(action_idx)

                    mcts.update_root(i, action_idx)

                    game_over = board.play_move(move)

                    if game_over:
                        game_over_count += 1
                        if game_over_count >= n_games:
                            break
                        boards[i] = Board(get_5_random_cards(cards = "standard"))

                        mcts.reset_tree(i)

            total_time = time.time() - start

            results_reuse[r] = total_time


            game_over_count = 0
            boards = [Board(get_5_random_cards(cards = "standard")) for _ in range(64)]
            for i in range(64):
                mcts.reset_tree(i)
            start = time.time()
            while game_over_count < n_games:
                policies = mcts.search_batch(boards)
                
                for i in range(64):
                    board = boards[i]
                    policy = policies[i]
                    
                    policy = F.relu(policy)
                    action_idx = torch.multinomial(policy, 1).item()
                    move = board.action_index_to_move(action_idx)

                    mcts.reset_tree(i)

                    game_over = board.play_move(move)

                    if game_over:
                        game_over_count += 1
                        if game_over_count >= n_games:
                            break
                        boards[i] = Board(get_5_random_cards(cards = "standard"))

            total_time = time.time() - start

            results_no_reuse[r] = total_time





        mean_batched = np.mean(results_reuse)
        std_batched = np.std(results_reuse)

        mean_unbatched = np.mean(results_no_reuse)
        std_unbatched = np.std(results_no_reuse)

        log[str(n_games)] = {
            "mean_reuse" : mean_batched,
            "std_reuse" : std_batched,
            "mean_no_reuse" : mean_unbatched,
            "std_no_reuse" : std_unbatched
        }
    
    pbar.close()

    json.dump(log, open(log_file, "w"), indent=4)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Number of simulations VS time")
    parser.add_argument("--config", "-c", type = str, default = "tree_reuse.yaml", 
                        help = "path to the config file")
    parser.add_argument("--output", "-o", type = str, default = "benchmark/output_tree_reuse.json", 
                        help = "path of the output")
    
    args = parser.parse_args()

    run_benchmark(args)