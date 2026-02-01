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
import re
from network.model import OnitamaNet
import json


class OnitamaStreamingDataset(IterableDataset):
    def __init__(self, files):
        self.files = files

    def __iter__(self):

        worker_info = get_worker_info()
        
        if worker_info is None:
            my_files = self.files
        else:
            per_worker = int(math.ceil(len(self.files) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.files))
            my_files = self.files[start:end]

        random.shuffle(my_files)

        for filepath in my_files:
            try:
                data_chunk = torch.load(filepath)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue

            random.shuffle(data_chunk)

            for raw_item in data_chunk:
                nn_input, policy_target, value_target = get_nn_training_data(raw_item)

                yield (
                    nn_input, 
                    torch.tensor(policy_target, dtype=torch.float32), 
                    torch.tensor(value_target, dtype=torch.float32)
                )
            
            del data_chunk



# data v0 is generated from a random model
# model v0 trains on data v0
# data v1 is generated from v0
# model v1 trains on data v1
def train(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = yaml.safe_load(open(os.path.join("models", "configs", args.config), "r"))

    model_name = config["model"]["name"]
    model_folder = os.path.join("models", "weights", model_name)
    data_folder = os.path.join("models", "data", model_name)
    logs_folder = os.path.join("models", "logs", model_name)

    os.makedirs("./models", exist_ok = True)
    os.makedirs("./models/configs", exist_ok = True)
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(data_folder, exist_ok=True)

    train_config = config["training"]
    batch_size = train_config.get('batch_size', 64)
    lr = train_config.get('learning_rate', 0.001)
    weight_decay = train_config.get('weight_decay', 1e-4)
    train_proportion = train_config.get('train_proportion', 0.9)
    train_workers = train_config.get('train_workers', 4)
    val_workers = train_config.get('val_workers', 2)
    total_steps = train_config.get('total_steps', 50000)
    test_steps = train_config.get('test_steps', 100)
    checkpoint_steps = train_config.get('checkpoint_steps', 1000)
    include_old_gens = train_config.get('include_old_gens', 5)

    # Finds the data corresponding to new_gen:

    # Data directories for each generation
    data_gens_dir = sorted(os.listdir(data_folder), key = lambda f : re.findall("[0-9]+", f)[0])

    if not data_gens_dir:
        raise(ValueError("No training data exists."))
    
    newest_data_gen_file = data_gens_dir[-1]
    newest_data_gen_idx = re.findall("[0-9]+", newest_data_gen_file)[0]

    # Take the last <included_dirs_indices> generations of data : sliding window
    included_dirs_indices = min(len(data_gens_dir), include_old_gens)
    old_data_gens = data_gens_dir[-included_dirs_indices:]
    
    # Extract all shards of data from these included data directories
    all_files = []
    for old_data_gen in old_data_gens:
        all_files.extend(os.listdir(old_data_gen))
        
    if not all_files:
        raise ValueError(f"No files found. Generate games before training !")
    
    # Train-test split
    random.shuffle(all_files)

    split_idx = int(len(all_files) * train_proportion)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    print(f"Training on {len(train_files)} shards. Validating on {len(val_files)} shards.")

    
    gen_key = f"v{newest_data_gen_idx}"

    # Load the model :
    model = OnitamaNet(config).to(device)

    # Look for the latest generation of models :
    # Naming convention for models : v5_40000.pt, where 5 is the generation and 40000 the number of steps.
    model_gens_files = sorted(os.listdir(model_folder), key = lambda f : re.findall("[0-9]+", f)[0])

    newest__model_gen_idx = 0
    steps = 0
    if not model_gens_files:
        print("No model has been found. Training v1 from randomly initialized weights.")


    else:
        newest_model_file = model_gens_files[-1]
        newest__model_gen_idx = re.findall("[0-9]+", newest_model_file)[0]
    
        # Keep training current generation if the model gen index matches the data gen index.
        if newest__model_gen_idx == newest_data_gen_idx:
            steps = re.findall("[0-9]+", newest_model_file)[1]
            print(f"Resuming training for generation v{newest__model_gen_idx} : steps {steps}/{total_steps}")
        
        # Otherwise, take the newest model and initialize a new one from its weights.
        else:
            print(f"Starting a new training for generation v{newest_data_gen_idx}. Model initialized from v{newest__model_gen_idx}")

        model.load_state_dict(torch.load(newest_model_file, weights_only = True))
        
    model_path = os.path.join(model_gens_files, gen_key)



    # Create datasets and loaders
    train_dataset = OnitamaStreamingDataset(train_files)
    val_dataset = OnitamaStreamingDataset(val_files)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,      
        num_workers=train_workers,     
        pin_memory=True,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,      
        num_workers=val_workers,     
        pin_memory=True,
        drop_last=True
    )


    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    value_criterion = nn.MSELoss()
    policy_criterion = nn.CrossEntropyLoss()


    # Log file
    log_file = os.path.join(logs_folder, model_name)

    if os.path.isfile(log_file):
        log = json.load(open(log_file, "r"))
    else:
        log = {}


    if log[gen_key] not in log:
        log[gen_key] = {}

    # Training loop
    while steps < total_steps:
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
            loss_v = value_criterion(v_pred.view(-1), target_v.view(-1))
            loss = loss_p + loss_v

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            p_loss_acc += loss_p.item()
            v_loss_acc += loss_v.item()
            batch_count += 1
            steps += 1

            if steps % checkpoint_steps == 0:
                checkpoint_path = model_path + f"_{steps}.pt"
                print(f"Saving checkpoint for v{newest_data_gen_idx} at {checkpoint_path}")
                torch.save({
                    'steps': steps,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)

            if steps % test_steps == 0:
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
                        l_v = value_criterion(v_pred.view(-1), target_v.view(-1))
                        val_loss_value += l_v.item()
                        val_loss_policy += l_p.item()
                        val_batch_count += 1

                avg_val_loss_policy = val_loss_policy / val_batch_count if val_batch_count > 0 else 0
                avg_val_loss_value = val_loss_value / val_batch_count if val_batch_count > 0 else 0

                print(f"Steps {steps}/{total_steps} :")
                print(f"Policy : {avg_val_loss_policy:.4f} | Value : {avg_val_loss_value:.4f}")

                log[gen_key][str(steps)] = {}
                log[gen_key][str(steps)]["Policy"] = avg_val_loss_policy
                log[gen_key][str(steps)]["Value"] = avg_val_loss_value

                json.dump(log, log_file, indent=4)



        checkpoint_path = model_path + f"_{steps}.pt"
        print(f"Saving checkpoint for v{newest_data_gen_idx} at {checkpoint_path}")
        torch.save({
            'steps': steps,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the network on a set of Onitama games.")
    parser.add_argument("--config", "-c", required=True, type = str, 
                        help = "Name of the configuration file in the .models/configs folder")

     
    args = parser.parse_args()

    train(args)