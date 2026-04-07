import json
import matplotlib.pyplot as plt
import argparse
import yaml
import os


def plot_json(args):
    config_path = os.path.join("benchmark", "configs", args.config)

    config = yaml.safe_load(open(config_path, "r"))

    output_path = config["out"]

    log_files = [log_file for log_file in config["logs"]]

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'gray', 'cyan', 'black', 'yellow'][:len(log_files)]

    fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (15, 4))
    axes = axes.flatten()

    for log_file, color in zip(log_files, colors):
        log = json.load(open(os.path.join("models", "logs", log_file), "r"))
        model_name = log_file.split(".")[0]
        gens = []
        pol_losses = []
        val_losses = []
        pol_accs = []
        wdl_losses = []
        
        for gen_key, gen_log in log.items():
            gens.append(int(gen_key.strip("v")))

            last_step_key = sorted(gen_log.keys(), key = lambda x : int(x))[-1]

            last_step_log = gen_log[last_step_key]

            pol_accs.append(last_step_log["Validation Policy Accuracy"])
            pol_losses.append(last_step_log["Validation Policy Loss"])
            val_losses.append(last_step_log["Validation Value MSE"])

            #if "Validation Value WDL" in last_step_log:
            #    wdl_losses.append(last_step_log["Validation Value WDL"])

        axes[0].plot(gens, pol_losses, color = color,label = model_name, linewidth = 2)
        axes[1].plot(gens, val_losses, color = color, label = model_name, linewidth = 2)
        axes[2].plot(gens, pol_accs, color = color, label = model_name, linewidth = 2)

        #if wdl_losses:
        #    axes[1].plot(gens, val_losses, color = color, linestyle = '-.', linewidth = 2)


    axes[0].set_xlabel("Generation index")
    axes[0].set_ylabel("Policy Loss")
    axes[0].grid(True, alpha = 0.7)
    axes[0].legend()
    
    axes[1].set_xlabel("Generation index")
    axes[1].set_ylabel("Value Loss")
    axes[1].grid(True, alpha = 0.7)
    axes[1].legend()

    axes[2].set_xlabel("Generation index")
    axes[2].set_ylabel("Policy Accuracy")
    axes[2].grid(True, alpha = 0.3)
    axes[2].legend()

    plt.tight_layout()

    plt.savefig(output_path)


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Number of simulations VS time")
    parser.add_argument("--config", "-c", type = str, default = "all.yaml", 
                        help = "path to the config file")
    
    args = parser.parse_args()

    plot_json(args)
