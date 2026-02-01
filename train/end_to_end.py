import argparse
from train.train import train
from train.generate import generate
import yaml
import os

def extract_model_gen_idx(str : str):
    try:
        idx = int(str.strip("v").split(".")[0].split("_")[0])
        return idx
    except:
        return -1
    
def does_exist_last_gen_model(num_gens, model_dir):
    for model_file in os.listdir(model_dir):
        gen = extract_model_gen_idx(model_file)

        if gen == num_gens:
            return True
        
    return False

def end_to_end(args):
    
    config = yaml.safe_load(open(os.path.join("models", "configs", args.config), "r"))
    num_gens = config.get("generations", 10)

    model_name = config["model"]["name"]
    model_dir = os.path.join("models", "weights", model_name)

    while not does_exist_last_gen_model(num_gens, model_dir):
        print("\n---------------------------------------------------------------")
        generate(args)
        print("\n---------------------------------------------------------------")
        train(args)

    print(f"Finished training model {model_name} until generation {num_gens}.")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End to end AlphaZero training \
                                     alternating between generation and supervised learning.")
    
    parser.add_argument("--config", "-c", required = True, type = str, 
                        help = "Name of the config file with the training hyperparameters.")
    
    args = parser.parse_args()

    end_to_end(args)