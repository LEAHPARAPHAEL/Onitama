import torch
from mcts.mcts_rollout import MCTS_Rollout
from network.model import OnitamaNet
from mcts.batched_mcts import BatchedMCTS
import os
import torch.nn.functional as F
import gzip

'''
Code for agentic play of Onitama
'''

class OnitamaAgent:
    def __init__(self, config, weights_path=None,base_path="./models", num_simulations=100):
        self.config_path = os.path.join(base_path, "configs")
        self.weight_path = os.path.join(base_path, "weights")
        self.active_model = None
        self.active_model_name = "None"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rollout = False
        self.active_config = config
        self.active_mcts = None

        self.loaded = self.setup_mcts(self.active_config, weights_path=weights_path,num_simulations=num_simulations)


    def setup_mcts(self, config, weights_path=None,num_simulations=100):
        print(f"Setting up MCTS with config: {config}")
        print(f"{config['model'].get('type',None)}")
        if config["model"].get('type',None) == "rollout":
            print("Using rollout MCTS")
            self.rollout = True
            self.active_mcts = MCTS_Rollout(config)
            self.active_model_name = config["model"]['name']
            return True
        else:
            try:
                self.active_config = config
                self.active_model = OnitamaNet(self.active_config).to(self.device)
                with gzip.open(weights_path, "rb") as f:
                    state_dict = torch.load(f, weights_only = False, map_location=self.device)
                model_state_dict = state_dict["model_state_dict"]
                self.active_model.load_state_dict(model_state_dict)
                self.active_model.eval()

                self.active_mcts = BatchedMCTS(self.active_model, self.active_config, self.device)
                self.active_mcts.num_simulations = num_simulations

                self.active_model_name = config["model"]['name']
                return True
            except Exception as e:
                print(f"Failed to load model: {e}")
        return False


    def select_move(self, board):
        if self.rollout:
            action_probs = self.active_mcts.search(board)
            policy = action_probs.clone().detach()
        else:
            policy = self.active_mcts.search_batch([board])[0]
            policy = F.relu(policy)

        action_idx = torch.multinomial(policy, 1).item()
        move = board.action_index_to_move(action_idx)
        return move