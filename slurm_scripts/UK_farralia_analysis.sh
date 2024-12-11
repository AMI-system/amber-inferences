#!/bin/bash
#SBATCH --job-name=process_test      # Job name
#SBATCH --output=./logs/temp.out
#SBATCH --partition=short-serial
#SBATCH --time=00:10:00
#SBATCH --mem=64
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=1                 # Number of CPU cores per task

# see https://help.jasmin.ac.uk/docs/batch-computing/slurm-queues/ from slurm help

source ~/miniforge3/bin/activate

# Define the path to your existing Conda environment (modify as appropriate)
CONDA_ENV_PATH="~/conda_envs/moth_detector_env/"

# Activate the environment
conda activate "${CONDA_ENV_PATH}"

# Print the Costa Rica deployments avaialble on the object store
# python print_deployments.py --subset_countries 'United Kingdom'

# Run the Inference script
python s3_download_with_inference.py --country "United Kingdom"  --deployment "Far Ralia - string" --keep_crops --crops_interval 10 --data_storage_path ./data/temp --regional_model_path ./models/turing-uk_v03_resnet50_2024-05-13-10-03_state.pt --regional_map_path ./models/03_uk_data_category_map.json --num_workers 4

echo "Pipeline completed successfully."
