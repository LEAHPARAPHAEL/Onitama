import torch
import math
from .mcts_node_rollout import MCTSNode_rollout

# from game.board import Board (Assuming this import exists)

class BatchedMCTS:
    def __init__(self, config, device):
        mcts_config = config['mcts_rollout']
        self.num_simulations = mcts_config.get('simulations', 100)

        self.high_temperature = mcts_config.get('high_temperature', 1.0)
        self.low_temperature = mcts_config.get('low_temperature', 0.0)

        self.lower_temperature_after = mcts_config.get('lower_temperature_after', 10)
        self.c_puct = mcts_config.get('c_puct', 1.0)

        mask_illegal_moves = config['training'].get('mask_illegal_moves', False)
        if mask_illegal_moves:
            self.policy_default = -1.0
        else:
            self.policy_default = 0.0

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

            if self.model.wdl:
                wdl_probs = torch.softmax(values, dim=1)
                values_scalar = wdl_probs[:, 2] - wdl_probs[:, 0]
                values_batch = values_scalar.cpu().numpy()

            else:
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

            probs = torch.full((1252,), self.policy_default, dtype=torch.float32)

            if boards[k].turn_count >= self.lower_temperature_after:
                temperature = self.low_temperature
            else:
                temperature = self.high_temperature

            if temperature == 0:
                best_action_idx = max(root.children, key=lambda k: root.children[k].visit_count)
                probs[best_action_idx] = 1.0

            else:
                total_visits = sum(c.visit_count ** temperature for c in root.children.values())
                
                assert(total_visits != 0)

                for action_idx, child in root.children.items():
                    probs[action_idx] = (child.visit_count ** temperature) / total_visits

                
            policies.append(probs)
            
        return policies