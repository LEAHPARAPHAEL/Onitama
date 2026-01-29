import numpy as np
from ..game.board import CARDS
import torch

def get_nn_input(board):
    """
    Converts bitboards to a (9, 5, 5) float32 tensor.
    Planes:
    0: Player disciples
    1: Player master
    2: Opponent disciples
    3: Opponent master
    4-5: Player cards
    6-7: Opponent cards
    8: Side card
    """
    
    planes = np.zeros((9, 5, 5), dtype=np.float32)
    

    def fill_plane(plane_idx, bitboard):
        for i in range(25):
            if (bitboard >> i) & 1:
                r, c = divmod(i, 5)
                planes[plane_idx, r, c] = 1.0

    fill_plane(0, board.my_disciples)
    fill_plane(1, board.my_master)
    fill_plane(2, board.opp_disciples)
    fill_plane(3, board.opp_master)
    
    def fill_card_pattern(plane_idx, card_id):
        offsets = CARDS[card_id] 
        center_r, center_c = 2, 2 
        planes[plane_idx, center_r, center_c] = 1.0 
        
        for dr, dc in offsets:
            nr, nc = center_r + dr, center_c + dc
            if 0 <= nr < 5 and 0 <= nc < 5:
                planes[plane_idx, nr, nc] = 1.0


    fill_card_pattern(4, board.my_cards[0])
    fill_card_pattern(5, board.my_cards[1])

    fill_card_pattern(6, board.opp_cards[0])
    fill_card_pattern(7, board.opp_cards[1])

    fill_card_pattern(8, board.side_card)

    return torch.tensor(planes).unsqueeze(0)