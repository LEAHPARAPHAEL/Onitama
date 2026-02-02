import torch
import copy
from .mcts_node import MCTSNode
import math
from network.input import get_nn_input
from game.board import Board

class MCTS:
    def __init__(self, model, config, device):
        self.model = model
        self.model.eval() 
        self.num_simulations = config.get('simulations', 800)
        self.temperature = config.get('temperature', 0.05)
        self.inverse_temperature = 1. / self.temperature
        self.c_puct = config.get('c_puct', 1.0)
        self.device = device
        
    def search(self, root_board : Board):
        """
        Runs MCTS simulations from the current root_board state.
        Returns the policy vector (probabilities) for the actual game move.
        """
        root = MCTSNode()
        
        for _ in range(self.num_simulations):
            simulation_board = root_board.clone()
            self._run_simulation(root, simulation_board)
            
        action_probs = torch.zeros(1252)
        
        total_visits = sum(child.visit_count ** self.inverse_temperature for child in root.children.values())
        
        for action_idx, child in root.children.items():
            action_probs[action_idx] = (child.visit_count ** self.inverse_temperature) / total_visits
            
        return action_probs

    def _run_simulation(self, node, board):
        """
        Recursive function to traverse, expand, and backpropagate.
        """
        while node.is_expanded and node.children:
            action_idx, node = node.select_child(self.c_puct)
            
            move = board.action_index_to_move(action_idx)
            game_over = board.play_move(move) 

            if game_over :
                result = board.get_result()
                node.backpropagate(result)
                return

        nn_input = get_nn_input(board).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.model(nn_input) 
        
        value = value.item()
        
        legal_moves = board.get_legal_moves()
        legal_indices = [board.move_to_action_index(m) for m in legal_moves]

        indices_tensor = torch.tensor(legal_indices, dtype=torch.long, device=logits.device)

        logits_vector = logits.squeeze(0)
        legal_logits = logits_vector[indices_tensor]
        
        legal_probs = torch.softmax(legal_logits, dim=0).tolist()
        
        valid_probs = dict(zip(legal_indices, legal_probs))
        
        node.expand(valid_probs)

        node.backpropagate(value)