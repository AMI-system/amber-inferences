#!/bin/bash

source ~/miniforge3/bin/activate
conda activate "~/conda_envs/flatbug/"

json_directory="./keys/costa_rica"
region="cri"
output_base_dir="./data/costa_rica"
credentials_file="./credentials.json"

# step 1: get the deployments
#python 01_print_deployments.py   --subset_countries 'Costa Rica' > ${json_directory}/costa_rica_deployments.txt
deployments=($(grep -oP 'Deployment ID: \Kdep[0-9]+' ${json_directory}/costa_rica_deployments.txt))

echo "Costa Rica deployments: ${deployments[@]}"

for deployment_id in "${deployments[@]}"; do
  echo "Processing keys for deployment: $deployment_id"

  json_file="${json_directory}/${deployment_id}_workload_chunks.json"

  # for each deployment get the keys
  # python 02_generate_keys.py \
  #   --bucket $region \
  #   --deployment_id $deployment_id \
  #   --output_file "${json_directory}/${deployment_id}_keys.txt"
  # python 03_pre_chop_files.py \
  #   --input_file "${json_directory}/${deployment_id}_keys.txt" \
  #   --file_extensions 'jpg' 'jpeg' \
  #   --chunk_size 100 \
  #   --output_file $json_file

  # if [ $? -ne 0 ]; then
  #   echo "Error getting keys for deployment $deployment_id"
  #   exit 1
  # fi



  mkdir -p ${output_base_dir}/${deployment_id}

  num_chunks=$(python3 -c "
import json
try:
  with open('$json_file') as f:
      data = json.load(f)
      print(len(data))
except Exception as e:
  print(f'Error reading $json_file: {e}', flush=True)
  exit(1)
")

  echo "Number of chunks: $num_chunks"

  if [ -z "$num_chunks" ] || ! [[ "$num_chunks" =~ ^[0-9]+$ ]]; then
    echo "Error: Invalid number of chunks for $json_file"
    continue
  fi

  sbatch --job-name="$deployment_id" \
        --gres gpu:1 \
        --partition orchid \
        --account orchid \
        --mem 8G \
        --array=1-$num_chunks \
        --export=ALL,json_file="$json_file",output_base_dir="$output_base_dir",deployment_id="$deployment_id",region="$region",credentials_file="$credentials_file" \
        ./slurm_scripts/costa_rica_sbatch.sh

  if [ $? -ne 0 ]; then
    echo "Error submitting job for chunk $SLURM_ARRAY_TASK_ID of deployment $deployment_id"
    exit 1
  fi

done











#   # python 05_combine_outputs.py \
#   #   --csv_file_pattern "${output_base_dir}/${deployment_id}/${deployment_id}_*.csv" \
#   #   --main_csv_file "${output_base_dir}/${deployment_id}.csv"


# done
