import matplotlib.pyplot as plt
import json
import argparse


def plot_json(args):

    json_path = args.data

    json_data = json.load(open(json_path, "r"))
    output_path = args.output

    sorted_keys = sorted(json_data.keys(), key=lambda x: int(x))

    x = [int(k) for k in sorted_keys]
    batched_y_mean = [json_data[k]['mean_batched'] for k in sorted_keys]
    batched_y_std = [json_data[k]['std_batched'] for k in sorted_keys]

    batched_y_upper = [m + s for m, s in zip(batched_y_mean, batched_y_std)]
    batched_y_lower = [m - s for m, s in zip(batched_y_mean, batched_y_std)]

    unbatched_y_mean = [json_data[k]['mean_unbatched'] for k in sorted_keys]
    unbatched_y_std = [json_data[k]['std_unbatched'] for k in sorted_keys]

    unbatched_y_upper = [m + s for m, s in zip(unbatched_y_mean, unbatched_y_std)]
    unbatched_y_lower = [m - s for m, s in zip(unbatched_y_mean, unbatched_y_std)]

    plt.figure(figsize=(6, 5))

    plt.plot(x, batched_y_mean, label='Batched', color='blue', marker='o', linewidth=1)

    plt.fill_between(x, batched_y_lower, batched_y_upper, color='blue', alpha=0.2)

    plt.plot(x, unbatched_y_mean, label='Sequential', color='red', marker='o', linewidth=1)

    plt.fill_between(x, unbatched_y_lower, unbatched_y_upper, color='red', alpha=0.2)

    plt.xlabel('Batch size')
    plt.ylabel('Forward pass time (s)')
    plt.title('Forward pass time for batched and sequential evaluation')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    plt.savefig(output_path)


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Number of simulations VS time")
    parser.add_argument("--data", "-d", type = str, default = "benchmark/output_batch_time.json", 
                        help = "path to the config file")
    parser.add_argument("--output", "-o", type = str, default = "plots/batch_time.png", 
                        help = "path of the output")
    
    args = parser.parse_args()

    plot_json(args)