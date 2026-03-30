#!/bin/bash
#SBATCH -p mesonet 
#SBATCH --account=m25146
#SBATCH --job-name=se_net_6_96_no_aug
#SBATCH --output=logs/se_net_6_96_no_aug.out
#SBATCH --error=logs/se_net_6_96_no_aug.out
#SBATCH --time=31:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --mail-user=oscar.garnier@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Activate virtual environment
source .venv/bin/activate

# Run the training script
python -m train.end_to_end -c se_net_6_96_no_aug.yaml
