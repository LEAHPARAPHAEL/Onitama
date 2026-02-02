from game.board import Board
from game.board_utils import CARDS, get_5_random_cards, Move
import random
import time
from network.input import get_nn_training_data
import torch

def rotate_180(b: int):
    """
    Bitwise rotation for 25-bit board.
    Reverses the bits of a 32-bit integer, then adjusts for the 25-bit size.
    """
    # 1. Standard 32-bit Reverse (Swap adjacent, then pairs, then nibbles...)
    b = ((b >> 1) & 0x55555555) | ((b & 0x55555555) << 1)
    b = ((b >> 2) & 0x33333333) | ((b & 0x33333333) << 2)
    b = ((b >> 4) & 0x0F0F0F0F) | ((b & 0x0F0F0F0F) << 4)
    b = ((b >> 8) & 0x00FF00FF) | ((b & 0x00FF00FF) << 8)
    b = (b >> 16) | (b << 16) & 0xFFFFFFFF

    # 2. Adjust for 25 bits
    # A 32-bit reverse puts the LSB (bit 0) at bit 31.
    # We want it at bit 24. So we shift right by (32 - 25) = 7.
    return b >> 7

def rotate_180_slow(bitboard : int):
    """
    Rotates the board so that the current player is always at the bottom.
    """
    return int(f"{bitboard:025b}"[::-1], 2)



def test():





    cards = get_5_random_cards()
    for card_id in cards:
        print(CARDS[card_id]["name"])
    board = Board(cards)

    legal_moves = board.get_legal_moves()

    random_move = random.choice(legal_moves)

    board.play_move(random_move)

    print(rotate_180(board.opponent_disciples))
    print(rotate_180_slow(board.opponent_disciples))
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