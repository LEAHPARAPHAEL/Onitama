import torch
import math
from .mcts_node import MCTSNode
from network.input import get_nn_input
# from game.board import Board (Assuming this import exists)

class BatchedMCTS:
    def __init__(self, model, config, device):
        self.model = model
        self.model.eval()
        self.num_simulations = config.get('simulations', 100)
        self.temperature = config.get('temperature', 0.05)
        self.inverse_temperature = 1. / self.temperature
        self.c_puct = config.get('c_puct', 1.0)
        self.device = device

    def search_batch(self, boards):
        """
        Returns a list of policies for all the boards in the list given as an argument.
        """
        
        batch_size = len(boards)

        roots = [MCTSNode() for _ in range(batch_size)]

        for _ in range(self.num_simulations):
            
            leaf_nodes = []
            leaf_boards = []
            valid_indices = []  
            nn_inputs = []

            for i in range(batch_size):

                if boards[i] is None:
                    continue

                node = roots[i]
                board = boards[i].clone() 
                
                while node.is_expanded and node.children:
                    action_idx, node = node.select_child(self.c_puct)
                    move = board.action_index_to_move(action_idx)
                    board.play_move(move) 

                    if board.is_game_over():
                        break
                    
                if board.is_game_over():
                    result = board.get_result() 
                    node.backpropagate(result)
                else:
                    leaf_nodes.append(node)
                    leaf_boards.append(board)
                    valid_indices.append(i)
                    nn_inputs.append(get_nn_input(board))

            if not valid_indices:
                continue

            input_batch = torch.stack(nn_inputs).to(self.device)
            
            with torch.no_grad():
                logits, values = self.model(input_batch)
            
            probs_batch = torch.softmax(logits, dim=1).cpu().numpy()
            values_batch = values.cpu().numpy()

            for j, game_idx in enumerate(valid_indices):
                node = leaf_nodes[j]
                board = leaf_boards[j]
                
                legal_moves = board.get_legal_moves()
                legal_indices = [board.move_to_action_index(m) for m in legal_moves]
                
                full_probs = probs_batch[j]
                
                valid_probs = {idx: full_probs[idx] for idx in legal_indices}
                total_p = sum(valid_probs.values())
                
                if total_p > 0:
                    valid_probs = {k: v / total_p for k, v in valid_probs.items()}
                else:
                    valid_probs = {k: 1.0/len(legal_indices) for k in legal_indices}

                val = values_batch[j].item()
                
                node.expand(valid_probs)
                node.backpropagate(val)

        policies = []
        for k, root in enumerate(roots):
            
            if boards[k] is None:
                policies.append(None)
                continue

            total_visits = sum(c.visit_count ** self.inverse_temperature for c in root.children.values())
            probs = torch.zeros(1252) 
            
            if total_visits > 0:
                for action_idx, child in root.children.items():
                    probs[action_idx] = (child.visit_count ** self.inverse_temperature) / total_visits
            else:
                probs[0] = 1.0 
                
            policies.append(probs)
            
        return policies