import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
import numpy as np
from network.input import get_nn_training_data
import argparse
import os
import yaml
import random
import math
from network.model import OnitamaNet
import json
from tqdm import tqdm
from game.board_utils import create_horizontal_flip_mask
import torch.nn.functional as F
from bisect import bisect_right
from train.train_utils import extract_gen_idx, extract_shard_idx, extract_model_steps, extract_positions
import shutil
import glob
import gzip
import sys

def extract_model_gen_idx(str : str):
    try:
        idx = int(str.strip("v").split(".")[0].split("_")[0])
        return idx
    except:
        return -1


class WDLValueLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        targets = targets.view(-1)
        target_indices = (targets + 1).long()
        
        return self.ce_loss(logits, target_indices)
    
class MSEValueLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, preds, targets):
        return self.mse_loss(preds.view(-1), targets.view(-1))
    
class MaskIllegalMovesPolicyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, p_logits, target_p):
        illegal_mask = target_p < -0.5
        clean_target = F.relu(target_p) 
        p_logits[illegal_mask] = -1.0e10
        return self.ce_loss(p_logits, clean_target)
    

def swap_player_cards(states, policies):

    new_states = states.clone()
    new_policies = policies.clone()

    card1_channel = states[..., 4, :, :].clone()
    card2_channel = states[..., 5, :, :].clone()

    new_states[..., 4, :, :] = card2_channel
    new_states[..., 5, :, :] = card1_channel

    new_policies[..., 0:625] = policies[..., 625:1250]
    new_policies[..., 625:1250] = policies[..., 0:625]
    
    new_policies[..., 1250] = policies[..., 1251]
    new_policies[..., 1251] = policies[..., 1250]

    return new_states, new_policies

def swap_opponent_cards(states):

    new_states = states.clone()

    card1_channel = states[..., 6, :, :].clone()
    card2_channel = states[..., 7, :, :].clone()

    new_states[..., 6, :, :] = card2_channel
    new_states[..., 7, :, :] = card1_channel

    return new_states


class OnitamaStreamingDataset(IterableDataset):
    def __init__(self, files, use_data_augmentation=False):
        self.files = files
        self.use_data_augmentation = use_data_augmentation
        if use_data_augmentation:
            self.flip_mask = create_horizontal_flip_mask()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            my_files = self.files
        else:
            per_worker = int(math.ceil(len(self.files) / float(worker_info.num_workers)))
            start = worker_info.id * per_worker
            end = min(start + per_worker, len(self.files))
            my_files = self.files[start:end]

        random.shuffle(my_files)

        for filepath in my_files:
            try:
                data = torch.load(filepath, weights_only=False)
                
                states = data["states"]   
                policies = data["policies"] 
                values = data["values"]  

                num_items = states.shape[0]


                indices = torch.randperm(num_items)

                for idx in indices:

                    nn_input = states[idx]
                    policy = policies[idx]
                    value = values[idx]

                    if self.use_data_augmentation:
                    
                        if torch.rand(1).item() > 0.5:
                            policy = policy[self.flip_mask]
                            nn_input = torch.flip(nn_input, dims=[-1])

                        if torch.rand(1).item() > 0.5:
                            nn_input, policy = swap_player_cards(nn_input, policy)

                        if torch.rand(1).item() > 0.5:
                            nn_input = swap_opponent_cards(nn_input)

                    yield nn_input, policy, value

            except Exception as e:
                print(f"Corrupt shard {filepath}: {e}")


def get_lr_lambda(warmup_steps, milestones, gamma=0.1):
    """
    Returns a lambda function for the scheduler.
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(warmup_steps)
        
        if not milestones:
            return 1.0
        
        index = bisect_right(milestones, current_step)
        return gamma ** index

    return lr_lambda




def shuffle_shards(files, shuffled_shards_dir, total_positions, 
                   max_positions_in_ram=100000, train_proportion=0.9):

    remove_buffer(shuffled_shards_dir)
        
    os.makedirs(shuffled_shards_dir, exist_ok=True)
    temp_dir = os.path.join(shuffled_shards_dir, "temp")
    train_dir = os.path.join(shuffled_shards_dir, "train")
    val_dir = os.path.join(shuffled_shards_dir, "val")
    
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    num_shards = (total_positions - 1) // max_positions_in_ram + 1
    
    def create_empty_buckets():
        return [{'states': [], 'policies': [], 'values': []} for _ in range(num_shards)]

    buckets = create_empty_buckets()
    positions_in_ram = 0
    flush_count = 0

    for filepath in tqdm(files, desc="Processing Inputs"):
        try:
            with gzip.open(filepath, "rb") as f:
                data = torch.load(f, weights_only=False)

            num_items = data["states"].shape[0]
            
            assignments = torch.randint(0, num_shards, (num_items,))

            for shard_idx in range(num_shards):
                mask = (assignments == shard_idx)
                
                if mask.any():
                    buckets[shard_idx]['states'].append(data['states'][mask])
                    buckets[shard_idx]['policies'].append(data['policies'][mask])
                    buckets[shard_idx]['values'].append(data['values'][mask])
            
            positions_in_ram += num_items
            
            if positions_in_ram >= max_positions_in_ram:
                _flush_buckets_to_disk(buckets, temp_dir, flush_count)
                buckets = create_empty_buckets()
                positions_in_ram = 0
                flush_count += 1
                
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    if positions_in_ram > 0:
        _flush_buckets_to_disk(buckets, temp_dir, flush_count)

    train_shards_paths = []
    val_shards_paths = []

    for shard_idx in tqdm(range(num_shards), desc="Finalizing Shards"):
        part_files = glob.glob(os.path.join(temp_dir, f"shard_{shard_idx}_part_*.pt"))
        if not part_files:
            continue
            
        all_states = []
        all_policies = []
        all_values = []
        
        for part in part_files:
            try:
                part_data = torch.load(part, weights_only=False)
                all_states.append(part_data['states'])
                all_policies.append(part_data['policies'])
                all_values.append(part_data['values'])
            except Exception as e:
                print(f"Error reading part {part}: {e}")
        
        if not all_states: continue
            
        merged_states = torch.cat(all_states)
        merged_policies = torch.cat(all_policies)
        merged_values = torch.cat(all_values)
        
        N = merged_states.shape[0]
        indices = torch.randperm(N)
        
        merged_states = merged_states[indices]
        merged_policies = merged_policies[indices]
        merged_values = merged_values[indices]

        split_idx = int(N * train_proportion)
        
        def save_split(s, p, v, path):
            if len(s) > 0:
                torch.save({
                    "states": s,
                    "policies": p,
                    "values": v
                }, path)
                return path
            return None

        t_path = save_split(
            merged_states[:split_idx], 
            merged_policies[:split_idx], 
            merged_values[:split_idx],
            os.path.join(train_dir, f"shard_{shard_idx}.pt")
        )
        
        v_path = save_split(
            merged_states[split_idx:], 
            merged_policies[split_idx:], 
            merged_values[split_idx:],
            os.path.join(val_dir, f"shard_{shard_idx}.pt")
        )

        if t_path: train_shards_paths.append(t_path)
        if v_path: val_shards_paths.append(v_path)

    shutil.rmtree(temp_dir)

    return train_shards_paths, val_shards_paths

def _flush_buckets_to_disk(buckets, temp_dir, flush_id):
    for i, bucket_data in enumerate(buckets):
        if len(bucket_data['states']) > 0:

            cat_states = torch.cat(bucket_data['states'])
            cat_policies = torch.cat(bucket_data['policies'])
            cat_values = torch.cat(bucket_data['values'])
            
            filename = os.path.join(temp_dir, f"shard_{i}_part_{flush_id}.pt")
            
            torch.save({
                "states": cat_states,
                "policies": cat_policies,
                "values": cat_values
            }, filename)
        

# data v0 is generated from a random model
# model v0 trains on data v0
# data v1 is generated from v0
# model v1 trains on data v1
def test(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = yaml.safe_load(open(os.path.join("models", "configs", args.config), "r"))

    model_name = config["model"]["name"]
    model_dir = os.path.join("models", "weights", model_name)
    data_dir = os.path.join("models", "data", model_name)
    log_file = os.path.join("models", "logs", f"{model_name}.json")

    remove_buffer(data_dir)

    os.makedirs("./models", exist_ok = True)
    os.makedirs("./models/configs", exist_ok = True)
    os.makedirs("./models/logs", exist_ok = True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    train_config = config["training"]
    batch_size = train_config.get('batch_size', 64)

    train_proportion = train_config.get('train_proportion', 0.9)
    val_workers = train_config.get('val_workers', 2)

    include_old_gens = train_config.get('include_old_gens', 5)

    wdl = config["model"].get("wdl", False)
    mask_illegal_moves = train_config.get("mask_illegal_moves", False)


    max_positions_in_ram = config["data"].get("max_positions_in_ram", 100000)

    # Finds the data corresponding to new_gen:

    data_gens_dir = sorted(os.listdir(data_dir), key = extract_gen_idx)


    if not data_gens_dir:
        raise(ValueError("No training data exists."))

    for model_file in os.listdir(model_dir):
        gen = extract_model_gen_idx(model_file)
        steps = extract_model_steps(model_file)

        data_gens = data_gens_dir[max(0, gen - include_old_gens + 1): gen + 1]
    
        # Extract all shards of data from these included data directories
        total_positions = 0
        all_files = []
        for old_data_gen_dir in data_gens:
            old_data_gen_shards = os.listdir(os.path.join(data_dir, old_data_gen_dir))
            for shard in old_data_gen_shards:
                all_files.append(os.path.join(data_dir, old_data_gen_dir, shard))
                total_positions += extract_positions(shard)
                
        if not all_files:
            raise ValueError(f"No files found. Generate games before training !")
        
        # Train-test split
        _, val_shards = shuffle_shards(all_files, os.path.join(data_dir, "buffer"), 
                                        total_positions = total_positions, 
                                        max_positions_in_ram=max_positions_in_ram,
                                        train_proportion=train_proportion)


        gen_key = f"v{gen}"


        # Load the model :
        model = OnitamaNet(config).to(device)

        with gzip.open(os.path.join(model_dir, model_file), "rb") as f:
            save_dict = torch.load(f, weights_only = False)

        model_state_dict = save_dict["model_state_dict"]
        model.load_state_dict(model_state_dict)
    
        # Create datasets and loaders

        print("Creating datasets.")
        val_dataset = OnitamaStreamingDataset(val_shards, use_data_augmentation=False)

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,    
            num_workers=val_workers,     
            pin_memory=True,
            drop_last=True,
            persistent_workers = True
        )



        value_criterion = nn.MSELoss()

        if wdl:
            value_criterion = WDLValueLoss()
        else:
            value_criterion = MSEValueLoss()

        if mask_illegal_moves:
            policy_criterion = MaskIllegalMovesPolicyLoss()
        else :
            policy_criterion = nn.CrossEntropyLoss()


        # Log file
        if os.path.isfile(log_file):
            log = json.load(open(log_file, "r"))
        else:
            log = {}


        if gen_key not in log:
            log[gen_key] = {}


        model.eval()
        val_loss_policy = 0
        val_loss_value = 0
        val_p_acc_correct = 0
        val_p_acc_total = 0
        val_batch_count = 0
        
        with torch.no_grad():
            for inputs, target_p, target_v in val_loader:
                inputs, target_p, target_v = inputs.to(device), target_p.to(device), target_v.to(device)
                p_logits, v_pred = model(inputs)
                l_p = policy_criterion(p_logits, target_p)
                l_v = value_criterion(v_pred, target_v)
                val_loss_value += l_v.item()
                val_loss_policy += l_p.item()
                val_p_acc_correct += (p_logits.argmax(dim=-1) == target_p.argmax(dim=-1)).sum().item()
                val_p_acc_total += inputs.size(0)
                val_batch_count += 1

        avg_val_loss_policy = val_loss_policy / val_batch_count if val_batch_count > 0 else 0
        avg_val_loss_value = val_loss_value / val_batch_count if val_batch_count > 0 else 0
        avg_val_p_acc = val_p_acc_correct / val_p_acc_total if val_p_acc_total > 0 else 0

        tqdm.write(f"\nPolicy : {avg_val_loss_policy:.4f} | Value : {avg_val_loss_value:.4f} | Acc : {avg_val_p_acc:.4f}")
        log[gen_key][str(steps)]["Val acc"] = avg_val_p_acc

        json.dump(log, open(log_file, "w"), indent=4)





        remove_buffer(data_dir)

def remove_buffer(data_dir):
    if os.path.exists(os.path.join(data_dir, "buffer")):
        shutil.rmtree(os.path.join(data_dir, "buffer"))


def remove_old_models(model_dir, gen_idx):
    for model_file in os.listdir(model_dir):
        if extract_gen_idx(model_file) == gen_idx:
            os.remove(os.path.join(model_dir, model_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the network on a set of Onitama games.")
    parser.add_argument("--config", "-c", required=True, type = str, 
                        help = "Name of the configuration file in the .models/configs folder")

     
    args = parser.parse_args()

    test(args)