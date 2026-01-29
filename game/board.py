import numpy as np
from collections import namedtuple

# Define a Move structure
Move = namedtuple('Move', ['from_idx', 'to_idx', 'card_id'])


CARDS = {
        0:  [(-2, 0), (1, 0)],           # Tiger (Blue)
        1:  [(-1, -2), (-1, 2), (1, -1), (1, 1)], # Dragon (Red)
        2:  [(-1, -1), (0, -2), (1, 1)],  # Frog (Blue)
        3:  [(-1, 1), (0, 2), (1, -1)],   # Rabbit (Red)
        4:  [(-1, 0), (0, -2), (0, 2)],   # Crab (Blue)
        5:  [(-1, -1), (-1, 1), (0, -1), (0, 1)], # Elephant (Red)
        6:  [(-1, -1), (0, -1), (0, 1), (1, 1)],  # Goose (Blue)
        7:  [(-1, 1), (0, -1), (0, 1), (1, -1)],  # Rooster (Red)
        8:  [(-1, -1), (-1, 1), (1, -1), (1, 1)], # Monkey (Blue)
        9:  [(-1, -1), (-1, 1), (1, 0)],  # Mantis (Red)
        10: [(-1, 0), (0, -1), (1, 0)],   # Horse (Blue)
        11: [(-1, 0), (0, 1), (1, 0)],    # Ox (Red)
        12: [(-1, 0), (1, -1), (1, 1)],   # Crane (Blue)
        13: [(-1, 0), (0, -1), (0, 1)],   # Boar (Red)
        14: [(-1, -1), (0, 1), (1, -1)],  # Eel (Blue)
        15: [(-1, 1), (0, -1), (1, 1)]    # Cobra (Red)
}

CARD_NAMES = [
    "Tiger", "Dragon", "Frog", "Rabbit", "Crab", "Elephant", "Goose", "Rooster",
    "Monkey", "Mantis", "Horse", "Ox", "Crane", "Boar", "Eel", "Cobra"
]

def get_card_str(card_id):
    """Returns a list of 5 strings representing the card's move pattern."""
    if card_id is None: return ["     "] * 5
    
    offsets = CARDS[card_id] # Uses the dict from previous step
    lines = []
    # Grid is 5x5, center is (2, 2)
    for r in range(5):
        line = ""
        for c in range(5):
            # Convert grid (r,c) to relative offset (dr, dc)
            # Row 0 is "Up 2" (dr = -2), Row 4 is "Down 2" (dr = +2)
            dr, dc = r - 2, c - 2
            
            if dr == 0 and dc == 0:
                line += " @ " # The Piece (Center)
            elif (dr, dc) in offsets:
                line += " X " # Valid Move
            else:
                line += " . "
        lines.append(line)
    return lines

'''
def precompute_move_tables(cards_dict):
    # Returns an array/list where index is card_id
    tables = [None] * 16
    for card_id, offsets in cards_dict.items():
        card_masks = []
        for i in range(25):
            mask = 0
            row, col = divmod(i, 5)
            for dr, dc in offsets:
                nr, nc = row + dr, col + dc
                if 0 <= nr < 5 and 0 <= nc < 5:
                    mask |= (1 << (nr * 5 + nc))
            card_masks.append(mask)
        tables[card_id] = card_masks
    return tables

MOVE_TABLES = precompute_move_tables(CARDS)
'''

MOVES = [
    [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32769, 65538, 131076, 262152, 524304, 1048608, 2097216, 4194432, 8388864, 16777728, 1024, 2048, 4096, 8192, 16384], 
    [64, 160, 320, 640, 256, 2052, 5128, 10257, 20482, 8196, 65664, 164096, 328224, 655424, 262272, 2101248, 5251072, 10503168, 20973568, 8392704, 131072, 262144, 557056, 65536, 131072], 
    [64, 128, 257, 514, 4, 2048, 4097, 8226, 16452, 136, 65536, 131104, 263232, 526464, 4352, 2097152, 4195328, 8423424, 16846848, 139264, 0, 32768, 1114112, 2228224, 4456448], 
    [4, 40, 80, 128, 256, 130, 1284, 2568, 4112, 8192, 4160, 41088, 82176, 131584, 262144, 133120, 1314816, 2629632, 4210688, 8388608, 4259840, 8519680, 17039360, 524288, 0],
    [4, 8, 17, 2, 4, 129, 258, 548, 72, 144, 4128, 8256, 17536, 2304, 4608, 132096, 264192, 561152, 73728, 147456, 4227072, 8454144, 17956864, 2359296, 4718592],
    [2, 5, 10, 20, 8, 66, 165, 330, 660, 264, 2112, 5280, 10560, 21120, 8448, 67584, 168960, 337920, 675840, 270336, 2162688, 5406720, 10813440, 21626880, 8650752],
    [66, 133, 266, 532, 8, 2112, 4257, 8514, 17028, 264, 67584, 136224, 272448, 544896, 8448, 2162688, 4359168, 8718336, 17436672, 270336, 2097152, 5275648, 10551296, 21102592, 8650752], 
    [2, 37, 74, 148, 264, 66, 1188, 2376, 4752, 8448, 2112, 38016, 76032, 152064, 270336, 67584, 1216512, 2433024, 4866048, 8650752, 2162688, 5373952, 10747904, 21495808, 8388608], 
    [64, 160, 320, 640, 256, 2050, 5125, 10250, 20500, 8200, 65600, 164000, 328000, 656000, 262400, 2099200, 5248000, 10496000, 20992000, 8396800, 65536, 163840, 327680, 655360, 262144], 
    [32, 64, 128, 256, 512, 1026, 2053, 4106, 8212, 16392, 32832, 65696, 131392, 262784, 524544, 1050624, 2102272, 4204544, 8409088, 16785408, 65536, 163840, 327680, 655360, 262144], 
    [32, 65, 130, 260, 520, 1025, 2082, 4164, 8328, 16656, 32800, 66624, 133248, 266496, 532992, 1049600, 2131968, 4263936, 8527872, 17055744, 32768, 1114112, 2228224, 4456448, 8912896], 
    [34, 68, 136, 272, 512, 1089, 2178, 4356, 8712, 16400, 34848, 69696, 139392, 278784, 524800, 1115136, 2230272, 4460544, 8921088, 16793600, 2129920, 4259840, 8519680, 17039360, 524288], 
    [64, 160, 320, 640, 256, 2049, 5122, 10244, 20488, 8208, 65568, 163904, 327808, 655616, 262656, 2098176, 5244928, 10489856, 20979712, 8404992, 32768, 65536, 131072, 262144, 524288], 
    [2, 5, 10, 20, 8, 65, 162, 324, 648, 272, 2080, 5184, 10368, 20736, 8704, 66560, 165888, 331776, 663552, 278528, 2129920, 5308416, 10616832, 21233664, 8912896], 
    [2, 36, 72, 144, 256, 64, 1153, 2306, 4612, 8200, 2048, 36896, 73792, 147584, 262400, 65536, 1180672, 2361344, 4722688, 8396800, 2097152, 4227072, 8454144, 16908288, 262144], 
    [64, 129, 258, 516, 8, 2050, 4132, 8264, 16528, 256, 65600, 132224, 264448, 528896, 8192, 2099200, 4231168, 8462336, 16924672, 262144, 65536, 1179648, 2359296, 4718592, 8388608]
]


class Board:

    GOAL_SQUARE_MASK = (1 << 2)

    def __init__(self, cards : list[int]):
        self.side_card           = cards[4]
        self.blue                = (self.side_card % 2 == 0)

        self.turn_count = 0

        self.player_disciples    = 0b1101100000000000000000000
        self.player_master       = 1 << 22
        self.opponent_disciples  = 0b11011
        self.opponent_master     = 1 << 2

        if self.blue:
            self.player_cards    = cards[0:2]
            self.opponent_cards  = cards[2:4]

        else:
            self.player_cards    = cards[2:4]
            self.opponent_cards  = cards[0:2]


    # Can probably be optimized a lot using pure bit manipulation, but fine for now.
    def rotate_180(self, bitboard : int ):
        """
        Rotates the board so that the current player is always at the bottom.
        """
        return int(f"{bitboard:025b}"[::-1], 2)


    def get_legal_moves(self):
        """
        Generates legal moves for the current player (at the bottom).

        Returns:
            list[Move]:
                All legal moves with the format Move(from_idx, to_idx, card_idx).
        """
        moves = []
        
        # All player pieces
        player_pieces = self.player_disciples | self.player_master

        # Iterates through the player's cards
        for card_id in self.player_cards:
            
            temp_pieces = player_pieces
            while temp_pieces:
                # Find the piece's location (Lowest Set Bit)
                from_idx = (temp_pieces & -temp_pieces).bit_length() - 1
                
                potential_destinations = MOVES[card_id][from_idx]
                
                valid_targets = potential_destinations & ~player_pieces
                
                
                while valid_targets:
                    to_idx = (valid_targets & -valid_targets).bit_length() - 1
                    moves.append(Move(from_idx, to_idx, card_id))
                    valid_targets &= (valid_targets - 1)
                    
                temp_pieces &= (temp_pieces - 1)
                
        return moves


    def switch_perspective(self):
        self.turn_count += 1
        # Rotate bitboards
        next_player_disciples   = self.rotate_180(self.opponent_disciples)
        next_player_master      = self.rotate_180(self.opponent_master)
        
        next_opponent_disciples = self.rotate_180(self.player_disciples)
        next_opponent_master    = self.rotate_180(self.player_master)
        
        # Apply the flip
        self.player_disciples   = next_player_disciples
        self.player_master      = next_player_master
        self.opponent_disciples = next_opponent_disciples
        self.opponent_master    = next_opponent_master
        
        # Swap cards
        self.player_cards, self.opponent_cards = self.opponent_cards, self.player_cards
        
        # Change turn
        self.blue ^= 1


    def play_move(self, move : Move):
        """
        Executes a move, checks for wins, swaps cards, and flips the board.
        
        Args:
            move (Move): A namedtuple(from_idx, to_idx, card_id)
        
        Returns:
            int: 
                0 if game continues
                1 if Current Player wins (You won)
                -1 if Opponent wins (Should verify illegal moves first, but standard safety)
        """

        from_mask = (1 << move.from_idx)
        to_mask   = (1 << move.to_idx)
        
        # Executes the move
        is_master_move = False
        
        if self.player_master & from_mask:
            # Master move
            self.player_master = (self.player_master ^ from_mask) | to_mask
            is_master_move = True
        else:
            # Disciple move
            self.player_disciples = (self.player_disciples ^ from_mask) | to_mask

        # Way of the Stone check
        captured_master = (self.opponent_master & to_mask) != 0
        
        # Capture
        self.opponent_disciples &= ~to_mask
        self.opponent_master    &= ~to_mask
        
        # Way of the Stone win
        if captured_master:
            self.turn_count += 1
            return 1
            
        # Way of the Stream win
        if is_master_move and (to_mask & Board.GOAL_SQUARE_MASK):
            self.turn_count += 1
            return 1

        # Takes the side card and replaces it with the played one
        self.player_cards.remove(move.card_id)
        self.player_cards.append(self.side_card)
        self.side_card = move.card_id

        self.switch_perspective()
        
        # Game continues
        return 0


    def move_to_action_index(self, move : Move):
        """
        Converts a Move tuple into a Neural Network action index (0-1249).
        """
        # Determine if this card is in slot 0 or slot 1 of the player's hand
        try:
            hand_slot = self.player_cards.index(move.card_id)
        except ValueError:
            raise ValueError(f"Card {move.card_id} not in hand {self.player_cards}")

        # Calculate the index
        # 25x25 = 625 moves per card
        # The index varies in this order [card_slot][from_idx][to_idx]
        # Index : (hand_slot * 625) + (from_idx * 25) + to_idx
        return (hand_slot * 625) + (move.from_idx * 25) + move.to_idx


    def action_index_to_move(self, action_index):
        """
        Converts a Neural Network action index (0-1249) back to a Move tuple.
        """
        hand_slot, remainder = divmod(action_index, 625)
        from_idx, to_idx = divmod(remainder, 25)
        
        card_id = self.player_cards[hand_slot]
        
        return Move(from_idx, to_idx, card_id)


    def __str__(self):
            if self.blue:
                blue_d, blue_m = self.player_disciples, self.player_master
                red_d, red_m   = self.opponent_disciples, self.opponent_master
                blue_cards, red_cards = self.player_cards, self.opponent_cards
                active_color = "BLUE (Bottom)"
            else:
                red_d = self.rotate_180(self.player_disciples)
                red_m = self.rotate_180(self.player_master)
                blue_d = self.rotate_180(self.opponent_disciples)
                blue_m = self.rotate_180(self.opponent_master)
                red_cards, blue_cards = self.player_cards, self.opponent_cards
                active_color = "RED (Top)"

            output = []
            output.append(f"\n=== Turn {self.turn_count}: {active_color} to play ===")
            
            output.append("Red Cards:")
            c1_lines = get_card_str(red_cards[0])
            c2_lines = get_card_str(red_cards[1])
            name1 = f"{CARD_NAMES[red_cards[0]]:^15}"
            name2 = f"{CARD_NAMES[red_cards[1]]:^15}"
            
            output.append(f"{name1}   {name2}")
            for l1, l2 in zip(c1_lines, c2_lines):
                output.append(f"{l1}   {l2}")
            output.append("-" * 40)

            side_card_lines = get_card_str(self.side_card)
            side_card_name = CARD_NAMES[self.side_card]

            board_rows = []
            for r in range(5):
                row_str = f"{r} "
                for c in range(5):
                    idx = r * 5 + c
                    mask = 1 << idx
                    
                    if blue_m & mask: char = " B " 
                    elif blue_d & mask: char = " b " 
                    elif red_m & mask: char = " R "
                    elif red_d & mask: char = " r " 
                    else: char = " . "
                    row_str += char
                board_rows.append(row_str)

            output.append(f"Side Card: {side_card_name}")
            for i in range(5):
                b_row = board_rows[i]
                s_row = side_card_lines[i]
                
                if not self.blue: 
                    output.append(f"{s_row}   |   {b_row}")
                else: 
                    output.append(f"{b_row}   |   {s_row}")

            output.append("-" * 40)
            output.append("Blue Cards:")
            c1_lines = get_card_str(blue_cards[0])
            c2_lines = get_card_str(blue_cards[1])
            name1 = f"{CARD_NAMES[blue_cards[0]]:^15}"
            name2 = f"{CARD_NAMES[blue_cards[1]]:^15}"
            
            for l1, l2 in zip(c1_lines, c2_lines):
                output.append(f"{l1}   {l2}")
            output.append(f"{name1}   {name2}")
            
            return "\n".join(output)