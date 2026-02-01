from network.model import OnitamaNet
import argparse
import torch
from mcts.batched_mcts import BatchedMCTS
import yaml
import os
import json
import itertools
from game.board_utils import get_5_cards_with_fixed_start
from tqdm import tqdm
from game.board import Board

def extract_model_gen_idx(str : str):
    try:
        idx = int(str.strip("v").split(".")[0].split("_")[0])
        return idx
    except:
        return -1

def tournament(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = yaml.safe_load(open(os.path.join("tournaments", "configs", args.config), "r"))

    tournament_config = config["tournament"]
    tournament_name = tournament_config["name"]

    # Should be divisible by 4
    rounds = tournament_config.get("rounds", 100)
    batch_size = tournament_config.get("batch_size", 64)
    half_round = rounds // 2
    quarter_round = rounds // 4


    models_config = config["models"]

    log_file = os.path.join("tournaments", "logs", f"{tournament_name}.json")
    if os.path.isfile(log_file):
        log = json.load(open(log_file, "r"))
    else:
        log = {}

    competitors = {}
    for model_name, model in models_config.items():
        model_config = model["config"]
        start_gen = model.get("start_gen", 0)
        end_gen = model.get("end_gen", 1000000)
        step_gen = model.get("step_gen", 1)

        model_dir = os.path.join("models", "weights", model_name)

        for model_gen_file in os.listdir(model_dir):
            gen = extract_model_gen_idx(model_gen_file)
            if gen >= start_gen and gen <= end_gen and (gen - start_gen) % step_gen == 0:
                competitor_name = f"{model_name}_v{gen}"
                competitors[competitor_name] = (os.path.join("models", "weights", model_name, model_gen_file), 
                                                os.path.join("models", "configs", model_config))
                if not competitor_name in log:
                    log[competitor_name] = {} 
        
    num_competitors = len(competitors)
    competitors_names = competitors.keys()
    pairs = list(itertools.combinations(competitors_names, 2))
    total_games = num_competitors * (num_competitors - 1) * rounds / 2

    pbar = tqdm(desc = "Total games in tournament", total = total_games)
    for c1, c2 in pairs:
        if c2 in log[c1] and c1 in log[c2] and "played" in log[c1][c2]:
            played = log[c1][c2]["played"]
            pbar.update(played)

        else:
            log[c1][c2] = {
                "played" : 0,
                "wins" : 0,
                "losses" : 0,
                "draws" : 0,

                "blue" : {
                    "first" : {
                        "wins" : 0,
                        "losses" : 0,
                        "draws" : 0,
                    },
                    "second" : {
                        "wins" : 0,
                        "losses" : 0,
                        "draws" : 0,
                    }
                },

                "red" : {
                    "first" : {
                        "wins" : 0,
                        "losses" : 0,
                        "draws" : 0,
                    },
                    "second" : {
                        "wins" : 0,
                        "losses" : 0,
                        "draws" : 0,
                    }
                }
            }

            log[c2][c1] = {
                "played" : 0,
                "wins" : 0,
                "losses" : 0,
                "draws" : 0,

                "blue" : {
                    "first" : {
                        "wins" : 0,
                        "losses" : 0,
                        "draws" : 0,
                    },
                    "second" : {
                        "wins" : 0,
                        "losses" : 0,
                        "draws" : 0,
                    }
                },

                "red" : {
                    "first" : {
                        "wins" : 0,
                        "losses" : 0,
                        "draws" : 0,
                    },
                    "second" : {
                        "wins" : 0,
                        "losses" : 0,
                        "draws" : 0,
                    }
                }
            }




    def play_quarter_round_between_engines(model_first,
                                           model_second,
                                           mcts_first, 
                                           mcts_second, 
                                           boards_quarter_round,
                                           first_color):
        boards = [None for _ in range(batch_size)]
        
        second_color = "red" if first_color == "blue" else "blue"
        idx = 0
        quarter_round_played_games = 0


        indices_to_replace = list(range(batch_size))
        idx = 0
        while quarter_round_played_games < quarter_round:

            for idx_to_replace in indices_to_replace:
                if idx < quarter_round:
                    boards[idx_to_replace] = boards_quarter_round[idx].clone()
                    idx += 1

            indices_to_replace = []
            policies_first = mcts_first.search_batch(boards)

            for i in range(batch_size):
                board = boards[i]
                if board is None:
                    continue

                policy = policies_first[i]
                
                action_idx = torch.multinomial(policy, 1).item()
                move = board.action_index_to_move(action_idx)

                game_over = board.play_move(move)

                if game_over:
                    log[model_first][model_second]["played"] += 1
                    log[model_second][model_first]["played"] += 1

                    result = board.get_result()
                    if result == -1:
                        log[model_first][model_second]["wins"] += 1
                        log[model_second][model_first]["losses"] += 1
                        log[model_first][model_second][first_color]["first"]["wins"] += 1
                        log[model_second][model_first][second_color]["second"]["losses"] += 1
                    else:
                        log[model_first][model_second]["draws"] += 1
                        log[model_second][model_first]["draws"] += 1
                        log[model_first][model_second][first_color]["first"]["draws"] += 1
                        log[model_second][model_first][second_color]["second"]["draws"] += 1

                    quarter_round_played_games += 1
                    pbar.update(1)

                    boards[i] = None
                    indices_to_replace.append(i)

            policies_second = mcts_second.search_batch(boards)

            for i in range(batch_size):
                board = boards[i]
                if board is None:
                    continue

                policy = policies_second[i]
                
                action_idx = torch.multinomial(policy, 1).item()
                move = board.action_index_to_move(action_idx)

                game_over = board.play_move(move)

                if game_over:
                    log[model_first][model_second]["played"] += 1
                    log[model_second][model_first]["played"] += 1

                    result = board.get_result()
                    if result == -1:
                        log[model_first][model_second]["losses"] += 1
                        log[model_second][model_first]["wins"] += 1
                        log[model_first][model_second][first_color]["first"]["losses"] += 1
                        log[model_second][model_first][second_color]["second"]["wins"] += 1
                    else:
                        log[model_first][model_second]["draws"] += 1
                        log[model_second][model_first]["draws"] += 1
                        log[model_first][model_second][first_color]["first"]["draws"] += 1
                        log[model_second][model_first][second_color]["second"]["draws"] += 1

                    quarter_round_played_games += 1
                    pbar.update(1)

                    boards[i] = None
                    indices_to_replace.append(i)

                

    for c1, c2 in pairs:
        tqdm.write(f"Starting match between {c1} and {c2}.")

        if log[c1][c2]["played"] == rounds:
            tqdm.write(f"Match already done. Moving to the next match.")
            continue

        weights1, config_file1 = competitors[c1]
        weights2, config_file2 = competitors[c2]

        config1 = yaml.safe_load(open(config_file1, "r"))
        config2 = yaml.safe_load(open(config_file2, "r"))

        model1 = OnitamaNet(config1).to(device)
        model2 = OnitamaNet(config2).to(device)

        save_dict1 = torch.load(weights1, weights_only = False)
        model_state_dict1 = save_dict1["model_state_dict"]
        model1.load_state_dict(model_state_dict1)

        save_dict2 = torch.load(weights2, weights_only = False)
        model_state_dict2 = save_dict2["model_state_dict"]
        model2.load_state_dict(model_state_dict2)


        blue_boards = [Board(get_5_cards_with_fixed_start(blue = True)) for _ in range(quarter_round)]
        red_boards = [Board(get_5_cards_with_fixed_start(blue = False)) for _ in range(quarter_round)]

        mcts1 = BatchedMCTS(model1, config1, device)
        mcts2 = BatchedMCTS(model2, config2, device)

        # Model 1 plays blue first
        play_quarter_round_between_engines(c1, c2, mcts1, mcts2, blue_boards, first_color = "blue")

        # Model 1 plays red first
        play_quarter_round_between_engines(c1, c2, mcts1, mcts2, red_boards, first_color = "red")

        # Model 2 plays blue first
        play_quarter_round_between_engines(c2, c1, mcts2, mcts1, blue_boards, first_color = "blue")

        # Model 2 plays red first
        play_quarter_round_between_engines(c2, c1, mcts2, mcts1, red_boards, first_color = "red")

        tqdm.write(f'Match statistics :\n \
                   {c1} -> [Wins : {log[c1][c2]["wins"]} | Draws : {log[c1][c2]["draws"]} | Losses : {log[c1][c2]["losses"]}]\n \
                    {c2} -> [Wins : {log[c2][c1]["wins"]} | Draws : {log[c2][c1]["draws"]} | Losses : {log[c2][c1]["losses"]}]')
        
        json.dump(log, open(log_file, "w"), indent=4)

    pbar.close()


        










        







if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play a tournament between multiple models.")
    parser.add_argument("--config", "-c", required=True, type = str, 
                        help = "Name of the configuration file in the .tournaments/configs/ directory.")

     
    args = parser.parse_args()

    tournament(args)