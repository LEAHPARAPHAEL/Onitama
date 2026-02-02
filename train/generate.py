import argparse
import torch
import yaml
import os
from network.model import OnitamaNet
from game.board_utils import get_5_random_cards
from game.board import Board
from mcts.batched_mcts import BatchedMCTS
import re
from pathlib import Path
from glob import glob
import shutil
import sys
from tqdm import tqdm

def extract_model_gen_idx(str : str):
    try:
        idx = int(str.strip("v").split(".")[0].split("_")[0])
        return idx
    except:
        return -1
    
def extract_shard_idx(str : str):
    try:
        idx = int(str.split("_")[-1].split(".")[0])
        return idx
    except:
        return -1
    
def extract_model_steps(str : str):
    try:
        idx = int(str.strip("v").split(".")[0].split("_")[-1])
        return idx
    except:
        return -1

def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = yaml.safe_load(open(os.path.join("models", "configs", args.config), "r"))


    model_dir = os.path.join("models", "weights", config["model"]["name"])
    data_dir = os.path.join("models", "data", config["model"]["name"])

    os.makedirs("./models", exist_ok = True)
    os.makedirs("./models/configs", exist_ok = True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    data_config = config["data"]
    mcts_config = config["mcts"]
    positions_per_shard = data_config.get("positions_per_shard", 5000)
    total_positions = data_config.get("total_positions", 50000)
    include_old_gens = config["training"].get("include_old_gens", 5)
    batch_size = data_config.get("batch_size", 64)
    total_steps = config["training"].get("total_steps", 50000)

    required_shards = total_positions // positions_per_shard

    # Load the model :
    model = OnitamaNet(config).to(device)
    # Look for the latest generation of models :
    model_gens_files = sorted(os.listdir(model_dir), key = extract_model_gen_idx)

    gen = 0
    if model_gens_files:
        newest_model_file = model_gens_files[-1]
        gen = extract_model_gen_idx(newest_model_file) + 1

        steps = extract_model_steps(newest_model_file)
        if steps < total_steps:
            print(f"Model v{gen - 1} has not finished training. Finish training before generating new positions.")
            return
        print(f"Found model for v{gen} generation. Loading from {newest_model_file}.")
        save_dict = torch.load(os.path.join(model_dir, newest_model_file), weights_only = False)
        model_state_dict = save_dict["model_state_dict"]
        model.load_state_dict(model_state_dict)

    else:
        print("No model has been found. Generating from a randomly initialized model (v0).")
        pass

    model.eval()

    # Look for data generated with the latest model :
    data_newest_gen_dir = os.path.join(data_dir, f"v{gen}")
    os.makedirs(data_newest_gen_dir, exist_ok=True)

    data_newest_gen_files = sorted(os.listdir(data_newest_gen_dir), key = extract_shard_idx)
    
    if data_newest_gen_files:
        last_shard = data_newest_gen_files[-1]
        last_shard_idx = extract_shard_idx(last_shard)
        if last_shard_idx == required_shards:
            print(f"All the positions for v{gen} have been generated.")
            return
        new_shard_idx = last_shard_idx + 1
    else:
        new_shard_idx = 0

    data = []
    data_path = os.path.join(data_newest_gen_dir, f"positions_{new_shard_idx}.pt")


    batched_mcts = BatchedMCTS(model, mcts_config, device)
    boards = [Board(get_5_random_cards()) for _ in range(batch_size)]
    game_histories = [[] for _ in range(batch_size)]

    positions = new_shard_idx * positions_per_shard

    pbar = tqdm(desc=f"v{gen} generation", total = total_positions)
    pbar.update(positions)
    while positions < total_positions:

        #print(f"Positions generated for v{gen} : {positions}/{total_positions}")

        policies = batched_mcts.search_batch(boards)
        for i in range(batch_size):
            board = boards[i]
            policy = policies[i]
            
            game_histories[i].append((
                board.get_compact_board(), 
                policy, 
                board.turn
            ))

        
            action_idx = torch.multinomial(policy, 1).item()
            move = board.action_index_to_move(action_idx)

            game_over = board.play_move(move)

            if game_over:
                result = board.get_result()
                potential_loser = board.get_turn()

                for compact_board, policy, turn in game_histories[i]:
                    value_label = (result if potential_loser == turn else -result)
                    data.append((compact_board, policy, value_label))
                    positions += 1
                    pbar.update(1)

                # Start a new game to replace the one that just finished
                boards[i] = Board(get_5_random_cards())
                
                
            while len(data) >= positions_per_shard:
                torch.save(data[:positions_per_shard], data_path)
                tqdm.write(f"\nSaving {positions_per_shard} positions to {data_path}.")
                new_shard_idx += 1
                data_path = os.path.join(data_newest_gen_dir, f"positions_{new_shard_idx}.pt")
                data = data[positions_per_shard:]

                if new_shard_idx == required_shards:
                    pbar.close()
                    remove_old_gens(data_dir, include_old_gens)
                    return






def remove_old_gens(data_dir, include_old_gens):
    # Remove old data generations
    sorted_data_dir = sorted(os.listdir(data_dir), key=lambda f : int(f.strip("v")))
    if len(sorted_data_dir) > include_old_gens:
        for old_data_gen_dir in sorted_data_dir[:-include_old_gens]:
            print(old_data_gen_dir)
            shutil.rmtree(os.path.join(data_dir, old_data_gen_dir))








if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a labeled set of Onitama games from a model.")
    parser.add_argument("--config", "-c", required=True, type = str, 
                        help = "Name of the configuration file in the .models/configs dir")

     
    args = parser.parse_args()

    generate(args)

