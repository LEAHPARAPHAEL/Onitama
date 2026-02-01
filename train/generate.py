import argparse
import torch
import yaml
import os
from network.model import OnitamaNet
from game.board_utils import get_5_random_cards
from game.board import Board
from mcts.mcts import MCTS
import re
from pathlib import Path
from glob import glob
import shutil



def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = yaml.safe_load(open(os.path.join("models", "configs", args.config), "r"))


    model_folder = os.path.join("models", "weights", config["model"]["name"])
    data_folder = os.path.join("models", "data", config["model"]["name"])

    os.makedirs("./models", exist_ok = True)
    os.makedirs("./models/configs", exist_ok = True)
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(data_folder, exist_ok=True)

    data_config = config["data"]
    mcts_config = config["mcts"]
    positions_per_file = data_config.get("positions_per_file", 5000)
    total_positions = data_config.get("total_positions", 20000)
    include_old_gens = config["training"].get("include_old_gens", 5)

    required_shards = total_positions // positions_per_file

    # Load the model :
    model = OnitamaNet(config).to(device)
    # Look for the latest generation of models :
    model_gens_files = sorted(os.listdir(model_folder), key = lambda f : re.findall("[0-9]+", f)[0])

    gen = 0
    if model_gens_files:
        newest_model_file = model_gens_files[-1]
        gen = re.findall("[0-9]+", newest_model_file)[0] + 1
        print(f"Found model for v{gen} generation. Loading from {newest_model_file}.")
        model.load_state_dict(torch.load(newest_model_file, weights_only = True))

    else:
        print("No model has been found. Generating from a randomly initialized model (v0).")

    model.eval()

    # Look for data generated with the latest model :
    data_newest_gen_dir = os.path.join(data_folder, f"v{gen}")
    os.makedirs(data_newest_gen_dir, exist_ok=True)

    data_newest_gen_files = sorted(os.listdir(data_newest_gen_dir), key = lambda f : re.findall("[0-9]+", f)[0])
    
    if data_newest_gen_dir:
        last_shard = data_newest_gen_files[-1]
        last_shard_idx = int(last_shard.split("_")[-1].split(".")[0])
        if last_shard_idx == required_shards:
            print(f"All the positions for v{gen} have been generated.")
            return
        new_shard_idx = last_shard_idx + 1
    else:
        new_shard_idx = 0

    data = []
    data_path = os.path.join(data_newest_gen_dir, f"positions_{new_shard_idx}.pt")


    mcts = MCTS(model, mcts_config, device)

    positions = new_shard_idx * positions_per_file

    while positions < total_positions:

        print(f"Positions generated for v{gen} : {positions}/{total_positions}")

        cards = get_5_random_cards()
        board = Board(cards)

        terminal = False

        game_history = []

        while not terminal:
            policy = mcts.search(board)

            game_history.append([
                board.clone(),
                policy
            ])

            action_idx = torch.multinomial(policy, 1).item()
            move = board.action_index_to_move(action_idx)
            terminal = board.play_move(move)

        # Result is either 0 or -1
        result = board.get_result()
        potential_loser = board.get_turn()

            
        for board, policy_label in game_history:
            # Remains 0 if the game is drawn, and alternates between -1 and 1 otherwise,
            # putting 1  when the turn is not that of the last player (the loser),
            # and 1 otherwise.
            value_label = (result if potential_loser == board.get_turn() else -result)
            data.append((board.get_compact_board(), policy_label, value_label))

        positions += len(game_history)

        if len(data) >= positions_per_file:
            torch.save(data[:positions_per_file], data_path)
            print(f"Saving {positions_per_file} positions to {data_path}.")
            new_shard_idx += 1
            data_path = os.path.join(data_newest_gen_dir, f"positions_{new_shard_idx}.pt")
            data = data[positions_per_file:]


    if data:
        remaining = total_positions - new_shard_idx*positions_per_file
    torch.save(data[:remaining], data_path)

    print(f"Saving {remaining} positions to {data_path}.")

    
    # Remove old data generations
    sorted_data_folder = sorted(os.listdir(data_folder), key=lambda f : int(f.strip("v")))
    if len(sorted_data_folder) > include_old_gens:
        for old_data_gen_folder in sorted_data_folder[:-include_old_gens]:
            shutil.rmtree(os.path.join(data_folder, old_data_gen_folder))












if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a labeled set of Onitama games from a model.")
    parser.add_argument("--config", "-c", required=True, type = str, 
                        help = "Name of the configuration file in the .models/configs folder")

     
    args = parser.parse_args()

    generate(args)

