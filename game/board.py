import numpy as np
from .board_utils import Move, CARDS, MOVES, get_card_str

class Board:

    GOAL_SQUARE_MASK = (1 << 2)
    SKIP_TURN_LEFT = Move(0, 0, 2)
    SKIP_TURN_RIGHT = Move(0, 1, 2)
    SKIP_TURN = [SKIP_TURN_LEFT, SKIP_TURN_RIGHT]

    def __init__(self, cards : list[int]):
        self.side_card           = cards[4]

        # 1 is Blue and 0 is Red
        self.turn                = (self.side_card % 2 == 0)

        self.turn_count = 0

        self.player_disciples    = 0b1101100000000000000000000
        self.player_master       = 1 << 22
        self.opponent_disciples  = 0b11011
        self.opponent_master     = 1 << 2

        if self.turn:
            self.player_cards    = cards[0:2]
            self.opponent_cards  = cards[2:4]

        else:
            self.player_cards    = cards[2:4]
            self.opponent_cards  = cards[0:2]

        self.result              = 0
        self.game_over           = False

        # Hashmap to store repetitions
        self.repetitions = {self.get_hash() : 1}


    def get_hash(self):
        """
        Returns a unique hash for the current board state.
        We sort the cards to ensure that holding (Card A, Card B) 
        hashes the same as holding (Card B, Card A).
        """
        player_cards = tuple(sorted(self.player_cards))
        opponent_cards = tuple(sorted(self.opponent_cards))

        state_tuple = (
            self.player_disciples,
            self.player_master,
            self.opponent_disciples,
            self.opponent_master,
            player_cards,
            opponent_cards,
            self.side_card,
            self.turn 
        )

        return hash(state_tuple)
    
    

    def clone(self):
        new_board = self.__class__.__new__(self.__class__)

        new_board.player_disciples   = self.player_disciples
        new_board.player_master      = self.player_master
        new_board.opponent_disciples = self.opponent_disciples
        new_board.opponent_master    = self.opponent_master

        new_board.player_cards       = self.player_cards[:]
        new_board.opponent_cards     = self.opponent_cards[:]
        new_board.side_card          = self.side_card

        new_board.turn               = self.turn
        new_board.turn_count         = self.turn_count 
        new_board.result             = self.result

        new_board.repetitions        = self.repetitions.copy()
        new_board.game_over          = self.game_over

        return new_board
    
    def get_result(self):
        return self.result
    
    def get_turn(self):
        return self.turn
    
    def is_game_over(self):
        return self.game_over
    
    def get_compact_board(self):
        """
        Returns a more lightweight representation of the board to store the position.
        """
        return (
            self.player_disciples,
            self.player_master,
            self.opponent_disciples,
            self.opponent_master,
            tuple(self.player_cards + self.opponent_cards + [self.side_card])
        )

    '''
    # Can probably be optimized a lot using pure bit manipulation, but fine for now.
    def rotate_180(self, bitboard : int):
        """
        Rotates the board so that the current player is always at the bottom.
        """
        return int(f"{bitboard:025b}"[::-1], 2)
    '''
    def rotate_180(self, b: int):
        """
        Rotates the board so that the current player is always at the bottom.
        Probably more efficient than the one above due to pure bitwise manipulation.
        """
        # Exchange pieces of the bitboard made of increasing length (1, 2, 4, 8, 16)
        b = ((b >> 1) & 0x55555555) | ((b & 0x55555555) << 1)
        b = ((b >> 2) & 0x33333333) | ((b & 0x33333333) << 2)
        b = ((b >> 4) & 0x0F0F0F0F) | ((b & 0x0F0F0F0F) << 4)
        b = ((b >> 8) & 0x00FF00FF) | ((b & 0x00FF00FF) << 8)
        b = (b >> 16) | (b << 16) & 0xFFFFFFFF

        # Shifts 7 to right to go from 32 bits to 25
        return b >> 7


    def get_legal_moves(self):
        """
        Generates legal moves for the current player (at the bottom).

        Returns:
            list[Move]:
                All legal moves with the format Move(from_idx, to_idx, card_slot).
        """
        moves = []

        # All player pieces
        player_pieces = self.player_disciples | self.player_master

        # Iterates through the player's cards
        for card_slot, card_id in enumerate(self.player_cards, 0):
            
            temp_pieces = player_pieces
            while temp_pieces:
                # Find the piece's location 
                from_idx = (temp_pieces & -temp_pieces).bit_length() - 1
                
                potential_destinations = MOVES[card_id][from_idx]
                
                valid_targets = potential_destinations & ~player_pieces
                
                
                while valid_targets:
                    to_idx = (valid_targets & -valid_targets).bit_length() - 1
                    moves.append(Move(from_idx, to_idx, card_slot))
                    valid_targets &= (valid_targets - 1)
                    
                temp_pieces &= (temp_pieces - 1)

        # Add skip turn if no legal moves otherwise
        if not moves:
            moves += Board.SKIP_TURN
                
        return moves


    def switch_perspective(self):
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
        self.turn ^= 1


    def play_move(self, move : Move):
        """
        Executes a move, checks for wins, swaps cards, and flips the board.
        
        Args:
            move (Move): A namedtuple(from_idx, to_idx, card_slot)
        
        Returns:
            int: 
                0 if game continues
                1 if game stops
        """
        self.turn_count += 1

        if move.card_slot < 2:
            from_mask = (1 << move.from_idx)
            to_mask   = (1 << move.to_idx)
            
            # Executes the move
            is_master_move = False
            
            if self.player_master & from_mask:
                # Master move
                self.player_master = to_mask
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
                self.result = -1
                
            # Way of the Stream win
            if is_master_move and (to_mask & Board.GOAL_SQUARE_MASK):
                self.result = -1

            # Takes the side card and replaces it with the played one
            played_card = self.player_cards.pop(move.card_slot)
            self.player_cards.append(self.side_card)
            self.side_card = played_card

        else:
            # If the move is to skip turn, the index is 1250 or 1251, so the 
            # card_slot = move_index // 625 = 2
            # from_idx = (move_index % 625) // 25 = 0
            # to_idx = (move_index % 625) % 25 = 0 or 1 (left or right)
            # So the card slot is indicated in to_idx

            # Takes the side card and replaces it with the played one
            played_card = self.player_cards.pop(move.to_idx)
            self.player_cards.append(self.side_card)
            self.side_card = played_card

        self.switch_perspective()
        
        current_hash = self.get_hash()
        count = self.repetitions.get(current_hash, 0) + 1
        self.repetitions[current_hash] = count

        # Game over
        if count >= 3 or self.result == -1:
            self.game_over = True
            return 1
        
        # Game continues
        return 0


    def move_to_action_index(self, move : Move):
        """
        Converts a Move tuple into a Neural Network action index (0-1249).
        """
        # Calculate the index
        # 25x25 = 625 moves per card
        # The index varies in this order [card_slot][from_idx][to_idx]
        # Index : (hand_slot * 625) + (from_idx * 25) + to_idx
        return (move.card_slot * 625) + (move.from_idx * 25) + move.to_idx


    def action_index_to_move(self, action_index):
        """
        Converts a Neural Network action index (0-1249) back to a Move tuple.
        """
        hand_slot, remainder = divmod(action_index, 625)
        from_idx, to_idx = divmod(remainder, 25)
        
        return Move(from_idx, to_idx, hand_slot)


    def __str__(self):
        if self.turn:
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
        c1_lines_raw = get_card_str(red_cards[0])
        c2_lines_raw = get_card_str(red_cards[1])
                    
        c1_lines = [line[::-1] for line in reversed(c1_lines_raw)]
        c2_lines = [line[::-1] for line in reversed(c2_lines_raw)]
        name1 = f'{CARDS[red_cards[0]]["name"]:^15}'
        name2 = f'{CARDS[red_cards[1]]["name"]:^15}'
        
        output.append(f"  {name1}     {name2}  ")
        for l1, l2 in zip(c1_lines, c2_lines):
            output.append(f"  {l1}     {l2}  ")
        output.append("-" * 40)

        side_card_lines = get_card_str(self.side_card)
        side_card_name = CARDS[self.side_card]["name"]

        board_rows = []
        for r in range(5):
            row_str = ""
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
            
            output.append(f"{s_row}       {b_row}")

        output.append("-" * 40)
        output.append("Blue Cards:")
        c1_lines = get_card_str(blue_cards[0])
        c2_lines = get_card_str(blue_cards[1])
        name1 = f'{CARDS[blue_cards[0]]["name"]:^15}'
        name2 = f'{CARDS[blue_cards[1]]["name"]:^15}'
        
        for l1, l2 in zip(c1_lines, c2_lines):
            output.append(f"  {l1}     {l2}  ")
        output.append(f"  {name1}     {name2}  ")
        
        return "\n".join(output)