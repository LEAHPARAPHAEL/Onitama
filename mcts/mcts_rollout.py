import torch
import copy
from .mcts_node_rollout import MCTSNode_Rollout
import math
import random
from network.input import get_nn_input
from game.board import Board

class MCTS_Rollout:
    def __init__(self, config):
        self.num_simulations = config.get('simulations', 800)
        self.temperature = config.get('temperature', 0.05)
        self.inverse_temperature = 1. / self.temperature
        self.c_puct = config.get('c_puct', 1.0)
        self.num_rollouts_per_simulation = config.get('num_rollouts_per_simulation', 100)
        self.num_nodes_to_explore_at_expansion = config.get('num_nodes_to_explore_at_expansion', 5)

    def search_batch(self, boards):
        '''
        Only for being compatible with the existing code 
        '''
        return [self.search(boards[0])]

    def search(self, root_board : Board):
        """
        Runs MCTS simulations from the current root_board state.
        Returns the policy vector (probabilities) for the actual game move.
        """
        root = MCTSNode_Rollout()
        
        for _ in range(self.num_simulations):
            simulation_board = root_board.clone()
            self._run_simulation(root, simulation_board)
            
        action_probs = torch.zeros(1252)
        
        total_visits = sum(child.visit_count ** self.inverse_temperature for child in root.children.values())
        
        for action_idx, child in root.children.items():
            action_probs[action_idx] = (child.visit_count ** self.inverse_temperature) / total_visits
            
        return action_probs

    def random_rollout(self,board : Board):
        current_player = board.turn
        rollout_board = board.clone()
        while True:
            if rollout_board.is_game_over():
                if rollout_board.turn == current_player:
                    ## current player lost
                    return rollout_board.get_result()
                else:
                    ## current player won
                    return -rollout_board.get_result()
            
            legal_moves = rollout_board.get_legal_moves()
            chosen_move = random.choice(legal_moves)
            rollout_board.play_move(chosen_move)

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

        #Expanding node with all valid moves
        legal_moves = board.get_legal_moves()
        legal_moves_idx = [board.move_to_action_index(move) for move in legal_moves]
        node.expand(legal_moves_idx)

        #Selecting a random move for rollout
        chosen_move = random.choice(legal_moves)
        child = node.children[board.move_to_action_index(chosen_move)]

        ## Evaluating move value
        board.play_move(chosen_move)
        avg_value = 0
        for i in range(self.num_rollouts_per_simulation):
            avg_value += self.random_rollout(board)
        avg_value /= self.num_rollouts_per_simulation

        child.backpropagate(avg_value)
