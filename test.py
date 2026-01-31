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
        print(f"\n> Executing Move: From {move.from_idx} to {move.to_idx} using {game.board.CARD_NAMES[board.player_cards[move.card_slot]]}")
        
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



import torch
import torch.nn as nn
import time
from network.model import OnitamaNet
import yaml

# --- IMPORTS (Replace these with your actual files) ---
# from game.board import Board
# from mcts.mcts import MCTS
# from network.model import AlphaZeroNet, get_nn_input

def play_test_game():
    print(">>> INITIALIZING SYSTEMS...")
    
    # 1. Setup Network (Random Weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = yaml.safe_load(open("configs/resnet_4_64.yaml", "r"))
    model = OnitamaNet(config).to(device)
    model.eval()
    print(f"Model created on {device}")

    # 2. Setup Config
    config = {
        'simulations': 50,  # Low number for fast testing
        'c_puct': 1.0
    }
    
    # 3. Initialize Board and MCTS
    # Assuming MCTS class is imported
    # mcts = MCTS(model, config) 
    # For this script to run standalone, I'll assume you have the MCTS class ready.
    # If not, paste your MCTS class here.
    
    from mcts.mcts_node import MCTSNode # Ensure this import works
    from mcts.mcts import MCTS
    # (Using the MCTS class code you provided earlier)
    all_cards = list(range(16))
    random.shuffle(all_cards)
    chosen_cards = all_cards[:5]
    
    # Initialize Board
    # (Assuming you put the class logic in 'OnitamaBoard')
    board = Board(chosen_cards)
    print(board)

    print("\n>>> STARTING GAME LOOP (Random AI vs Random AI)")
    
    move_count = 0
    
    game_over = False
    while not game_over:
        move_count += 1
        start_time = time.time()
        
        # --- A. Run MCTS ---
        # We need a fresh MCTS instance or reset tree usually, 
        # but for simple testing, creating a new one is safest/easiest.
        mcts = MCTS(model, config, device) 
        action_probs = mcts.search(board) # Returns tensor of shape (1250,)
        
        # --- B. Pick Action ---
        # For testing, we can just pick the Argmax (Best Move)
        # or Sample from distribution (to see variety)
        best_action_idx = torch.argmax(action_probs).item()
        

        move = board.action_index_to_move(best_action_idx)
        game_over = board.play_move(move)
        
        # --- D. Stats ---
        duration = time.time() - start_time
        print(f"Time: {duration:.2f}s | Max Prob: {action_probs[best_action_idx]:.4f}")
        print(board)
        print("-" * 40)

        # Safety break to prevent infinite loops if logic is broken
        if move_count > 100:
            print("!!! EMERGENCY STOP: Game exceeded 100 moves. Check Repetition Logic.")
            break

    # --- E. Game Over ---
    print("\n>>> GAME OVER")
    print(f"Total Moves: {move_count}")
    print(f"Final Result (from current perspective): {board.get_result()}")

if __name__ == "__main__":
    #run_random_game()
    play_test_game()