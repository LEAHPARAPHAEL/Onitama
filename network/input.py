import numpy as np
from game.board_utils import CARDS
import torch

def get_nn_input(board):

    planes = np.zeros((9, 5, 5), dtype=np.float32)
    

    def fill_plane(plane_idx, bitboard):
        for i in range(25):
            if (bitboard >> i) & 1:
                r, c = divmod(i, 5)
                planes[plane_idx, r, c] = 1.0

    fill_plane(0, board.player_disciples)
    fill_plane(1, board.player_master)
    fill_plane(2, board.opponent_disciples)
    fill_plane(3, board.opponent_master)
    
    def fill_card_pattern(plane_idx, card_id):
        offsets = CARDS[card_id]["pattern"]
        center_r, center_c = 2, 2 
        planes[plane_idx, center_r, center_c] = 1.0 
        
        for dr, dc in offsets:
            nr, nc = center_r + dr, center_c + dc
            if 0 <= nr < 5 and 0 <= nc < 5:
                planes[plane_idx, nr, nc] = 1.0


    fill_card_pattern(4, board.player_cards[0])
    fill_card_pattern(5, board.player_cards[1])

    fill_card_pattern(6, board.opponent_cards[0])
    fill_card_pattern(7, board.opponent_cards[1])

    fill_card_pattern(8, board.side_card)

    return torch.tensor(planes).unsqueeze(0)


def get_nn_training_data(data):
    """
    Returns a tuple with the network input, the policy label and the value label.
    """

    compact_board, policy_label, value_label = data

    player_disciples, player_master, opponent_disciples, opponent_master, cards = compact_board

    planes = np.zeros((9, 5, 5), dtype=np.float32)
    

    def fill_plane(plane_idx, bitboard):
        for i in range(25):
            if (bitboard >> i) & 1:
                r, c = divmod(i, 5)
                planes[plane_idx, r, c] = 1.0

    fill_plane(0, player_disciples)
    fill_plane(1, player_master)
    fill_plane(2, opponent_disciples)
    fill_plane(3, opponent_master)
    
    def fill_card_pattern(plane_idx, card_id):
        offsets = CARDS[card_id]["pattern"]
        center_r, center_c = 2, 2 
        planes[plane_idx, center_r, center_c] = 1.0 
        
        for dr, dc in offsets:
            nr, nc = center_r + dr, center_c + dc
            if 0 <= nr < 5 and 0 <= nc < 5:
                planes[plane_idx, nr, nc] = 1.0


    fill_card_pattern(4, cards[0])
    fill_card_pattern(5, cards[1])

    fill_card_pattern(6, cards[2])
    fill_card_pattern(7, cards[3])

    fill_card_pattern(8, cards[4])

    return torch.tensor(planes), policy_label, value_label
   

