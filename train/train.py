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


class OnitamaStreamingDataset(IterableDataset):
    def __init__(self, files, wdl):
        self.files = files
        self.wdl = wdl

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
                data_chunk = torch.load(filepath, weights_only=False)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue

            random.shuffle(data_chunk)

            for raw_item in data_chunk:
                nn_input, policy_target, value_target = get_nn_training_data(raw_item)

                yield (
                    nn_input, 
                    policy_target.clone().detach(),
                    torch.tensor(value_target, dtype=torch.float32)
                )
            
            del data_chunk

def extract_data_gen_idx(str : str):
    try:
        idx = int(str.strip("v").split(".")[0])
        return idx
    except:
        return -1

def extract_model_gen_idx(str : str):
    try:
        idx = int(str.strip("v").split(".")[0].split("_")[0])
        return idx
    except:
        return -1
    
def extract_model_steps(str : str):
    try:
        idx = int(str.strip("v").split(".")[0].split("_")[-1])
        return idx
    except:
        return -1
    
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

    os.makedirs("./models", exist_ok = True)
    os.makedirs("./models/configs", exist_ok = True)
    os.makedirs("./models/logs", exist_ok = True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    train_config = config["training"]
    batch_size = train_config.get('batch_size', 64)
    lr = train_config.get('learning_rate', 0.001)
    weight_decay = train_config.get('weight_decay', 1e-4)
    train_proportion = train_config.get('train_proportion', 0.9)
    train_workers = train_config.get('train_workers', 4)
    val_workers = train_config.get('val_workers', 2)
    total_steps = train_config.get('total_steps', None)
    test_steps = train_config.get('test_steps', 100)
    checkpoint_steps = train_config.get('checkpoint_steps', 1000)
    include_old_gens = train_config.get('include_old_gens', 5)
    momentum = train_config.get("momentum", 0.9)
    nesterov = train_config.get("nesterov", True)

    


    # Finds the data corresponding to new_gen:

    # Data directories for each generation
    data_gens_dir = sorted(os.listdir(data_dir), key = extract_data_gen_idx)

    if not data_gens_dir:
        raise(ValueError("No training data exists."))
    
    newest_data_gen_dir = data_gens_dir[-1]
    newest_data_gen_idx = extract_data_gen_idx(newest_data_gen_dir)

    # Take the last <included_dirs_indices> generations of data : sliding window
    included_dirs_indices = min(len(data_gens_dir), include_old_gens)
    old_data_gens = data_gens_dir[-included_dirs_indices:]
    
    # Extract all shards of data from these included data directories
    all_files = []
    for old_data_gen_dir in old_data_gens:
        old_data_gen_shards = os.listdir(os.path.join(data_dir, old_data_gen_dir))
        for shard in old_data_gen_shards:
            all_files.append(os.path.join(data_dir, old_data_gen_dir, shard))
        
    if not all_files:
        raise ValueError(f"No files found. Generate games before training !")
    
    # Train-test split
    random.shuffle(all_files)

    split_idx = int(len(all_files) * train_proportion)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    print(f"Training on {len(train_files)} shard(s). Validating on {len(val_files)} shard(s).")

    
    gen_key = f"v{newest_data_gen_idx}"



    # Compute number of steps in total
    # Total steps overrides the number of epochs
    # If not specified, infers the number of training steps required
    if not total_steps:
        epochs = train_config.get("epochs", 10)
        total_positions = config["data"].get("total_positions", 100000)
        total_steps = epochs * total_positions * len(old_data_gens) // batch_size


    # Load the model :
    model = OnitamaNet(config).to(device)

    # Optimizer
    #optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)

    # Milestones are in number of epochs
    lr_schedule = train_config.get("lr_schedule", None)
    milestones = []
    if lr_schedule:
        for milestone in lr_schedule:
            milestones.append(milestone)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = milestones, gamma = 0.1)

    # Look for the latest generation of models :
    # Naming convention for models : v5_40000.pt, where 5 is the generation and 40000 the number of steps.
    model_gens_files = sorted(os.listdir(model_dir), key = extract_model_gen_idx)

    newest_model_gen_idx = 0
    steps = 0
    if not model_gens_files:
        print("No model has been found. Training v0 from randomly initialized weights.")

    else:
        newest_model_file = model_gens_files[-1]
        newest_model_gen_idx = extract_model_gen_idx(newest_model_file)
    
        # Keep training current generation if the model gen index matches the data gen index.
        if newest_model_gen_idx == newest_data_gen_idx:
            steps = extract_model_steps(newest_model_file)
            print(f"Resuming training for generation v{newest_model_gen_idx} : steps {steps}/{total_steps}")
        
        # Otherwise, take the newest model and initialize a new one from its weights.
        else:
            print(f"Starting a new training for generation v{newest_data_gen_idx}. Model initialized from v{newest_model_gen_idx}")

        save_dict = torch.load(os.path.join(model_dir, newest_model_file), weights_only = False)
        model_state_dict = save_dict["model_state_dict"]
        optimizer_state_dict = save_dict["optimizer_state_dict"]
        scheduler_state_dict = save_dict["scheduler_state_dict"]
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        scheduler.load_state_dict(scheduler_state_dict)
        
    model_path = os.path.join(model_dir, gen_key)


    # Create datasets and loaders
    train_dataset = OnitamaStreamingDataset(train_files, model.wdl)
    val_dataset = OnitamaStreamingDataset(val_files, model.wdl)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,  
        num_workers=train_workers,     
        pin_memory=True,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,    
        num_workers=val_workers,     
        pin_memory=True,
        drop_last=True
    )


    value_criterion = nn.MSELoss()

    if model.wdl:
        value_criterion = nn.CrossEntropyLoss()
    else:
        value_criterion = nn.MSELoss()
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
            loss_v = value_criterion(v_pred.view(-1), target_v.view(-1))
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
                        l_v = value_criterion(v_pred.view(-1), target_v.view(-1))
                        val_loss_value += l_v.item()
                        val_loss_policy += l_p.item()
                        val_batch_count += 1

                avg_val_loss_policy = val_loss_policy / val_batch_count if val_batch_count > 0 else 0
                avg_val_loss_value = val_loss_value / val_batch_count if val_batch_count > 0 else 0

                tqdm.write(f"\nValidation : steps {steps}/{total_steps} \nPolicy : {avg_val_loss_policy:.4f} | Value : {avg_val_loss_value:.4f}")

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
                    'scheduler_state_dict': scheduler.state_dict()
                }, checkpoint_path)

            if steps == total_steps:
                break

        if steps == total_steps:
            break
        
        scheduler.step()

    pbar.close()



def remove_old_models(model_dir, gen_idx):
    for model_file in os.listdir(model_dir):
        if extract_model_gen_idx(model_file) == gen_idx:
            os.remove(os.path.join(model_dir, model_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the network on a set of Onitama games.")
    parser.add_argument("--config", "-c", required=True, type = str, 
                        help = "Name of the configuration file in the .models/configs folder")

     
    args = parser.parse_args()

    train(args)