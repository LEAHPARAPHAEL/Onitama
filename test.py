from game.board import Board
from game.board_utils import CARDS, get_5_random_cards, Move
import random
import time
from network.input import get_nn_training_data, get_nn_input
import torch
from network.model import OnitamaNet
import os
import yaml


def create_horizontal_flip_mask():
    mask = torch.arange(1252, dtype=torch.long)

    num_cards = 2
    board_size = 5
    squares = board_size * board_size  
    stride_card = squares * squares    
    stride_from = squares             

    def get_flipped_sq(sq_idx):
        """Convert linear 0-24 index to (row, col), flip col, return new index."""
        row = sq_idx // board_size
        col = sq_idx % board_size
        
        new_col = (board_size - 1) - col  
        
        return row * board_size + new_col


    for card_idx in range(num_cards):
        for from_sq in range(squares):
            for to_sq in range(squares):
                

                original_idx = (card_idx * stride_card) + \
                               (from_sq * stride_from) + \
                               to_sq

                new_from = get_flipped_sq(from_sq)
                new_to = get_flipped_sq(to_sq)

                new_idx = (card_idx * stride_card) + \
                          (new_from * stride_from) + \
                          new_to
                
                mask[original_idx] = new_idx

    return mask



def test():

    #cards = get_5_random_cards()
    #for card_id in cards:
    #    print(CARDS[card_id]["name"])

    cards = list(range(0, 5))
    
    board = Board(cards)

    compact_board = board.get_compact_board()

    policy = torch.zeros(1252)

    policy[1] = 1.0

    value_label = 1.0

    data = get_nn_training_data((compact_board, policy, value_label))

    network_input, policy_label, value_label = data

    



if __name__ == "__main__":
    test()

    '''
    # --- Usage    ---
    flip_mask = create_horizontal_flip_mask()

    # Verify a test case: 
    # Move from (0,0) to (0,4) [Top-Left to Top-Right] 
    # Should become (0,4) to (0,0) [Top-Right to Top-Left]
    idx_original = (0 * 625) + (0 * 25) + 4   # From 0 to 4
    idx_flipped  = (0 * 625) + (4 * 25) + 0   # From 4 to 0

    print(f"Original Index: {idx_original}")     # 4
    print(f"Mapped Index:   {flip_mask[idx_original]}") # 100 (4*25 + 0)
    assert flip_mask[idx_original] == idx_flipped
    print("Test passed!")

    board = Board(get_5_random_cards())

    random_idx = random.randint(0, 1252)

    print(board.action_index_to_move(random_idx))
    print(board.action_index_to_move(flip_mask[random_idx].item()))
    '''


