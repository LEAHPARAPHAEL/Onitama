from game.board import Board
import game.board
import random
import time

def run_random_game():
    # 1. Setup: 5 Random cards
    all_cards = list(range(16))
    random.shuffle(all_cards)
    chosen_cards = all_cards[:5]
    
    # Initialize Board
    # (Assuming you put the class logic in 'OnitamaBoard')
    board = Board(chosen_cards)
    
    print("Starting Random Game...")
    print(board)
    time.sleep(1)
    
    while board.turn_count < 100:
        # 1. Generate Legal Moves
        moves = board.get_legal_moves()
        
        if not moves:
            print(f"No moves available! {board.turn_count} Draw?")
            break
            
        # 2. Pick a Random Move
        move = random.choice(moves)
        print(f"\n> Executing Move: From {move.from_idx} to {move.to_idx} using {game.board.CARD_NAMES[move.card_id]}")
        
        # 3. Apply Move & Check Win
        result = board.play_move(move)
        
        # 4. Display New State
        print(board)
        
        # 5. Handle Game Over
        if result == 1:
            winner = "BLUE" if (board.blue) else "RED"
            # Note: We check % 2 != 0 because turn_count incremented inside play_move
            print(f"\n GAME OVER! {winner} Wins!")
            return
            
        time.sleep(0.5) # Slight pause to watch the game

# Run it!
run_random_game()