import math
import numpy as np

class MCTSNode:
    def __init__(self, parent=None, prior=0.0):
        self.parent = parent
        self.children = {}  
        
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior 
        
        self.is_expanded = False
        
    @property
    def value(self):
        """Returns the mean Action Value Q(s,a)."""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_child(self, c_puct=1.0):
        """
        Selects the best child according to the PUCT formula.
        """
        best_score = -float('inf')
        best_action = -1
        best_child = None

        # Precompute sqrt of total visits for efficiency
        sqrt_total_visits = math.sqrt(self.visit_count)

        for action_idx, child in self.children.items():
            # Q value (exploit)
            q_value = child.value
            
            # U value (explore): c_puct * P(s,a) * (sqrt(N_parent) / (1 + N_child))
            u_value = c_puct * child.prior * (sqrt_total_visits / (1 + child.visit_count))
            
            # Logic for unvisited nodes:
            # If a node is unvisited, Q is 0. 
            # The Prior determines order of exploration for unvisited nodes.
            
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_action = action_idx
                best_child = child

        return best_action, best_child

    def expand(self, policy_probs):
        """
        Expands the node by creating children for all valid actions.
        policy_probs: A dictionary {action_index: probability} 
                      already masked and normalized by the MCTS driver.
        """
        self.is_expanded = True
        for action_idx, prob in policy_probs.items():
            if prob > 0:
                self.children[action_idx] = MCTSNode(parent=self, prior=prob)

    def backpropagate(self, value):
        """
        Update stats and recurse up to the root.
        value: The evaluation of the game state from the perspective of the 
               player who just moved to reach this node.
        """
        self.visit_count += 1
        self.value_sum += value
        
        # Take the opposite of the value to give to the parent
        # (Because the turn has changed)
        if self.parent:
            self.parent.backpropagate(-value)