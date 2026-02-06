#!/usr/bin/env python3
import argparse
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "remote/scripts/template.sh"
CONFIGS = ROOT / "models/configs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job", help="config/job name without .yaml")
    args = ap.parse_args()

    cfg_path = CONFIGS / f"{args.job}.yaml"
    data = yaml.safe_load(cfg_path.read_text())
    cr = data.get("compute_resources", {})

    text = TEMPLATE.read_text().replace("template", args.job)
    lines = []
    for line in text.splitlines():
        if line.startswith("#SBATCH --time=") and "hours" in cr:
            line = f"#SBATCH --time={int(cr['hours']):02d}:00:00"
        elif line.startswith("#SBATCH --ntasks=") and "ntask" in cr:
            line = f"#SBATCH --ntasks={cr['ntask']}"
        elif line.startswith("#SBATCH --cpus-per-task=") and "cpus" in cr:
            line = f"#SBATCH --cpus-per-task={cr['cpus']}"
        elif line.startswith("#SBATCH --mem=") and "memory" in cr:
            line = f"#SBATCH --mem={cr['memory']}G"
        elif line.startswith("#SBATCH --gres=gpu:") and "gpus" in cr:
            line = f"#SBATCH --gres=gpu:{cr['gpus']}"
        lines.append(line)

    out_path = ROOT / "remote/scripts" / f"{args.job}.sh"
    out_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
