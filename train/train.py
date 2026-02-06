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

                    if self.use_data_augmentation and torch.rand(1).item() > 0.5:
                        policy = policy[self.flip_mask]
                        nn_input = torch.flip(nn_input, dims=[-1])

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
            data = torch.load(filepath, weights_only=False)

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
def train(args):

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
    lr = train_config.get('learning_rate', 0.01)
    weight_decay = train_config.get('weight_decay', 1e-4)
    train_proportion = train_config.get('train_proportion', 0.9)
    train_workers = train_config.get('train_workers', 4)
    val_workers = train_config.get('val_workers', 2)
    total_steps = train_config.get('total_steps', None)
    test_steps = train_config.get('test_steps', None)
    checkpoint_steps = train_config.get('checkpoint_steps', None)
    include_old_gens = train_config.get('include_old_gens', 5)
    momentum = train_config.get("momentum", 0.9)
    nesterov = train_config.get("nesterov", True)
    use_data_augmentation = train_config.get("use_data_augmentation", False)

    wdl = config["model"].get("wdl", False)
    mask_illegal_moves = train_config.get("mask_illegal_moves", False)
    warmup_steps = train_config.get("warmup_steps", 1000)
    lr_schedule = train_config.get("lr_schedule", None)

    max_positions_in_ram = config["data"].get("max_positions_in_ram", 100000)

    # Finds the data corresponding to new_gen:

    # Data directories for each generation
    data_gens_dir = sorted(os.listdir(data_dir), key = extract_gen_idx)

    if not data_gens_dir:
        raise(ValueError("No training data exists."))
    
    newest_data_gen_dir = data_gens_dir[-1]
    newest_data_gen_idx = extract_gen_idx(newest_data_gen_dir)

    # Take the last <included_dirs_indices> generations of data : sliding window
    included_dirs_indices = min(len(data_gens_dir), include_old_gens)
    old_data_gens = data_gens_dir[-included_dirs_indices:]
    
    # Extract all shards of data from these included data directories
    total_positions = 0
    all_files = []
    for old_data_gen_dir in old_data_gens:
        old_data_gen_shards = os.listdir(os.path.join(data_dir, old_data_gen_dir))
        for shard in old_data_gen_shards:
            all_files.append(os.path.join(data_dir, old_data_gen_dir, shard))
            total_positions += extract_positions(shard)
            
    if not all_files:
        raise ValueError(f"No files found. Generate games before training !")
    
    # Train-test split
    train_shards, val_shards = shuffle_shards(all_files, os.path.join(data_dir, "buffer"), 
                                     total_positions = total_positions, 
                                     max_positions_in_ram=max_positions_in_ram,
                                     train_proportion=train_proportion)


    gen_key = f"v{newest_data_gen_idx}"



    # Compute number of steps in total
    # Total steps overrides the number of epochs
    # If not specified, infers the number of training steps required
    if not total_steps:
        epochs = train_config.get("epochs", 10)
        total_steps = epochs * total_positions // batch_size
        # If total steps is not specified, the warmup steps should be in terms of epochs
        # This converts them to steps
        if lr_schedule:
            lr_schedule = list(map(lambda epoch: epoch * total_positions // batch_size, lr_schedule))

    if not checkpoint_steps:
        checkpoint_steps = total_steps // 10

    if not test_steps:
        test_steps = total_steps // 10


    # Load the model :
    model = OnitamaNet(config).to(device)

    # Optimizer
    #optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)


    lr_lambda = get_lr_lambda(warmup_steps=warmup_steps, milestones=lr_schedule) 
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lr_lambda)

    # Look for the latest generation of models :
    # Naming convention for models : v5_40000.pt, where 5 is the generation and 40000 the number of steps.
    model_gens_files = sorted(os.listdir(model_dir), key = extract_gen_idx)

    newest_model_gen_idx = 0
    steps = 0
    if not model_gens_files:
        print("No model has been found. Training v0 from randomly initialized weights.")

    else:
        newest_model_file = model_gens_files[-1]
        newest_model_gen_idx = extract_gen_idx(newest_model_file)
        save_dict = torch.load(os.path.join(model_dir, newest_model_file), weights_only = False)
        model_state_dict = save_dict["model_state_dict"]
        model.load_state_dict(model_state_dict)
    
        # Keep training current generation if the model gen index matches the data gen index.
        if newest_model_gen_idx == newest_data_gen_idx:
            steps = extract_model_steps(newest_model_file)
            print(f"Resuming training for generation v{newest_model_gen_idx} : steps {steps}/{total_steps}")
            optimizer_state_dict = save_dict["optimizer_state_dict"]
            scheduler_state_dict = save_dict["scheduler_state_dict"]
            
            optimizer.load_state_dict(optimizer_state_dict)
            scheduler.load_state_dict(scheduler_state_dict)
        
        # Otherwise, take the newest model and initialize a new one from its weights.
        else:
            print(f"Starting a new training for generation v{newest_data_gen_idx}. Model initialized from v{newest_model_gen_idx}")

        
    model_path = os.path.join(model_dir, gen_key)


    # Create datasets and loaders

    print("Creating datasets.")
    train_dataset = OnitamaStreamingDataset(train_shards, use_data_augmentation=use_data_augmentation)
    val_dataset = OnitamaStreamingDataset(val_shards, use_data_augmentation=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,  
        num_workers=train_workers, 
        pin_memory = True,    
        drop_last=True,
        persistent_workers = True
    )

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

    # Training loop
    pbar = tqdm(desc=f"v{newest_data_gen_idx} training", total = total_steps)
    pbar.update(steps)
    while True:
        # Train
        model.train()
        total_loss = 0
        p_loss_acc = 0
        v_loss_acc = 0
        batch_count = 0

        for inputs, target_p, target_v in train_loader:
            inputs, target_p, target_v = inputs.to(device), target_p.to(device), target_v.to(device)

            optimizer.zero_grad()
            
            p_logits, v_pred = model(inputs)

            loss_p = policy_criterion(p_logits, target_p)
            loss_v = value_criterion(v_pred, target_v)
            loss = loss_p + loss_v

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            p_loss_acc += loss_p.item()
            v_loss_acc += loss_v.item()
            batch_count += 1
            steps += 1
            pbar.update(1)

            if steps % test_steps == 0 or (steps == total_steps and total_steps % test_steps != 0):
                tqdm.write(f"\nTrain : steps {steps}/{total_steps} \nPolicy : {p_loss_acc / batch_count:.4f} | Value : {v_loss_acc / batch_count:.4f}")
                # Validation
                model.eval()
                val_loss_policy = 0
                val_loss_value = 0
                val_batch_count = 0
                
                with torch.no_grad():
                    for inputs, target_p, target_v in val_loader:
                        inputs, target_p, target_v = inputs.to(device), target_p.to(device), target_v.to(device)
                        
                        p_logits, v_pred = model(inputs)
                        
                        l_p = policy_criterion(p_logits, target_p)
                        l_v = value_criterion(v_pred, target_v)
                        val_loss_value += l_v.item()
                        val_loss_policy += l_p.item()
                        val_batch_count += 1

                avg_val_loss_policy = val_loss_policy / val_batch_count if val_batch_count > 0 else 0
                avg_val_loss_value = val_loss_value / val_batch_count if val_batch_count > 0 else 0

                tqdm.write(f"\nValidation : steps {steps}/{total_steps} \nPolicy : {avg_val_loss_policy:.4f} | Value : {avg_val_loss_value:.4f}")
                tqdm.write(f"\nLearning rate : {scheduler.get_last_lr()}")

                log[gen_key][str(steps)] = {}
                log[gen_key][str(steps)]["Train policy"] = p_loss_acc / batch_count
                log[gen_key][str(steps)]["Train value"] = v_loss_acc / batch_count
                log[gen_key][str(steps)]["Val policy"] = avg_val_loss_policy
                log[gen_key][str(steps)]["Val value"] = avg_val_loss_value

                json.dump(log, open(log_file, "w"), indent=4)

            if steps % checkpoint_steps == 0 or (steps == total_steps and total_steps % checkpoint_steps != 0):
                remove_old_models(model_dir, newest_data_gen_idx)
                checkpoint_path = model_path + f"_{steps}.pt"
                tqdm.write(f"\nSaving checkpoint for v{newest_data_gen_idx} at {checkpoint_path}")
                torch.save({
                    'steps': steps,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'training_over': steps == total_steps
                }, checkpoint_path)

            scheduler.step()

            if steps == total_steps:
                break
        

        if steps == total_steps:
            break
    

    pbar.close()

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

    train(args)