from game.board import Board
from game.board_utils import CARDS, get_5_random_cards, Move
import random
import time
from network.input import get_nn_training_data, get_nn_input
import torch
from network.model import OnitamaNet
import os
import yaml


def test():

    #cards = get_5_random_cards()
    #for card_id in cards:
    #    print(CARDS[card_id]["name"])

    cards = list(range(0, 5))
    
    board = Board(cards)

    boards = []
    while not board.is_game_over():
        legal_moves = board.get_legal_moves()
        random_move = random.choice(legal_moves)
        boards.append((board.clone(), random_move))
        board.play_move(random_move)

    last_board, last_move = boards[-1]
    last_action_idx = last_board.move_to_action_index(last_move)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network_input = get_nn_input(last_board).unsqueeze(0).to(device)

    config = yaml.safe_load(open(os.path.join("models", "configs", "resnet_4_64.yaml"), "r"))
    model = OnitamaNet(config).to(device)
    save_dict = torch.load(os.path.join("models", "weights", "resnet_4_64", "v0_50000.pt"), weights_only = False)
    model_state_dict = save_dict["model_state_dict"]
    model.load_state_dict(model_state_dict)
    policy, value = model(network_input)

    print(value)
    print(policy.squeeze(0)[last_action_idx])
    '''

    compact_board = board.get_compact_board()

    policy = torch.zeros(1252)
    policy[0] = 1.0

    value_label = 1.0

    data = get_nn_training_data((compact_board, policy, value_label))

    network_input, policy_label, value_label = data

    print(network_input)
    print(policy_label)
    print(value_label)
    '''



if __name__ == "__main__":
    test()