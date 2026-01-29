import torch
import copy
from mcts_node import MCTSNode
import math
from ..network.input import get_nn_input

class MCTS:
    def __init__(self, model, config):
        self.model = model
        self.model.eval() 
        self.num_simulations = config.get('mcts_simulations', 800)
        self.c_puct = config.get('c_puct', 1.0)
        
    def search(self, root_board):
        """
        Runs MCTS simulations from the current root_board state.
        Returns the policy vector (probabilities) for the actual game move.
        """
        # 1. Create Root
        root = MCTSNode()
        
        # 2. Run Simulations
        for _ in range(self.num_simulations):
            simulation_board = copy.deepcopy(root_board)
            self._run_simulation(root, simulation_board)
            
        action_probs = torch.zeros(1250)
        
        total_visits = sum(child.visit_count for child in root.children.values())
        
        for action_idx, child in root.children.items():
            action_probs[action_idx] = child.visit_count / total_visits
            
        return action_probs

    def _run_simulation(self, node, board):
        """
        Recursive function to traverse, expand, and backpropagate.
        """
        while node.is_expanded and node.children:
            action_idx, node = node.select_child(self.c_puct)
            
            move = board.action_index_to_move(action_idx)
            board.play_move(move) 

        legal_moves = board.get_legal_moves()
        
        if not legal_moves:
            # Terminal State: Current player has no moves (Loss) 
            # or we need specific win check logic if your engine allows wins 
            # that don't empty the move list (like Stream win).
            # Assuming play_move returns 1 on win, we might need to store 
            # 'game_over' status in the board.
            
            # Simple fallback: if no moves, it's a loss for current player.
            # Value = -1 (Loss)
            node.backpropagate(-1) 
            return

        nn_input = get_nn_input(board) 
        
        with torch.no_grad():
            log_policy, value = self.model(nn_input)
            
        value = value.item() 
        
        valid_probs = {}
        policy_sum = 0
        
        for move in legal_moves:
            idx = board.move_to_action_index(move)
            prob = math.exp(log_policy[0, idx].item())
            valid_probs[idx] = prob
            policy_sum += prob
            
        if policy_sum > 0:
            for idx in valid_probs:
                valid_probs[idx] /= policy_sum
        else:
            # Fallback if NN predicts 0 for all legal moves (rare numerical issue)
            # Distribute uniformly
            uniform = 1.0 / len(valid_probs)
            for idx in valid_probs:
                valid_probs[idx] = uniform
                
        node.expand(valid_probs)
        
        node.backpropagate(value)