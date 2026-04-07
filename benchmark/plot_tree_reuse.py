import matplotlib.pyplot as plt
import json
import argparse


def plot_json(args):

    json_path = args.data

    json_data = json.load(open(json_path, "r"))
    output_path = args.output

    sorted_keys = sorted(json_data.keys(), key=lambda x: int(x))

    x = [int(k) for k in sorted_keys]
    batched_y_mean = [json_data[k]['mean_reuse'] for k in sorted_keys]
    batched_y_std = [json_data[k]['std_reuse'] for k in sorted_keys]

    batched_y_upper = [m + s for m, s in zip(batched_y_mean, batched_y_std)]
    batched_y_lower = [m - s for m, s in zip(batched_y_mean, batched_y_std)]

    unbatched_y_mean = [json_data[k]['mean_no_reuse'] for k in sorted_keys]
    unbatched_y_std = [json_data[k]['std_no_reuse'] for k in sorted_keys]

    unbatched_y_upper = [m + s for m, s in zip(unbatched_y_mean, unbatched_y_std)]
    unbatched_y_lower = [m - s for m, s in zip(unbatched_y_mean, unbatched_y_std)]

    plt.figure(figsize=(6, 5))

    plt.plot(x, batched_y_mean, label='Tree reuse', color='blue', marker='o', linewidth=1)

    plt.fill_between(x, batched_y_lower, batched_y_upper, color='blue', alpha=0.2)

    plt.plot(x, unbatched_y_mean, label='No tree reuse', color='red', marker='o', linewidth=1)

    plt.fill_between(x, unbatched_y_lower, unbatched_y_upper, color='red', alpha=0.2)

    plt.xlabel('Number of games')
    plt.ylabel('Generation time (s)')
    plt.title('Generation time with and without tree reuse')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    plt.savefig(output_path)


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Number of simulations VS time")
    parser.add_argument("--data", "-d", type = str, default = "benchmark/output_tree_reuse.json", 
                        help = "path to the config file")
    parser.add_argument("--output", "-o", type = str, default = "plots/tree_reuse.png", 
                        help = "path of the output")
    
    args = parser.parse_args()

    plot_json(args)