import torch
import copy
from .mcts_node_rollout import MCTSNode_Rollout
import math
import random
from network.input import get_nn_input
from game.board import Board

class MCTS_Rollout:
    def __init__(self, config):
        mcts_config = config['mcts']
        self.num_simulations = mcts_config.get('simulations', 100)

        self.high_temperature = mcts_config.get('high_temperature', 1.0)
        self.low_temperature = mcts_config.get('low_temperature', 0.0)

        self.lower_temperature_after = mcts_config.get('lower_temperature_after', 10)
        self.c_puct = mcts_config.get('c_puct', 1.0)


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
        
        if root_board.turn_count >= self.lower_temperature_after:
            temperature = self.low_temperature
        else:
            temperature = self.high_temperature

        if temperature == 0:
            best_action_idx = max(root.children, key=lambda k: root.children[k].visit_count)
            action_probs[best_action_idx] = 1.0

        else:
            total_visits = sum(c.visit_count ** temperature for c in root.children.values())
            
            assert(total_visits != 0)

            for action_idx, child in root.children.items():
                action_probs[action_idx] = (child.visit_count ** temperature) / total_visits
            
        return action_probs

    def random_rollout(self,board : Board):
        while True:
            if board.is_game_over():
                return board.get_result()
            
            legal_moves = board.get_legal_moves()
            chosen_move = random.choice(legal_moves)
            board.play_move(chosen_move)

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

        legal_moves = board.get_legal_moves()

        chosen_move = random.choice(legal_moves)

        child = node.expand(board.move_to_action_index(chosen_move))
        board.play_move(chosen_move)
        value = self.random_rollout(board)
        child.backpropagate(value)

