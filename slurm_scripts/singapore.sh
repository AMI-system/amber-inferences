#!/bin/bash

source ~/miniforge3/bin/activate
conda activate "~/conda_envs/flatbug/"

json_directory="./keys/singapore"
region="sgp"
output_base_dir="./data/singapore"
credentials_file="./credentials.json"

# amber-keys --bucket $region' --deployment_id 'dep000082' --output_file './keys/singapore/dep000082_keys_subset.json' --subset_dates '2024-08-01' '2024-08-02'

# get all json files in the directory
all_json_files=(${json_directory}/dep*_workload_chunks.json)

# # remove the ones containing dep000045, 46, 51 (already run)
# all_json_files=(${all_json_files[@]//*dep000045*/})
# all_json_files=(${all_json_files[@]//*dep000046*/})
# all_json_files=(${all_json_files[@]//*dep000051*/})

# subset to only those containing dep000045, 56, 51
all_json_files=($(printf '%s\n' "${all_json_files[@]}" |sed -E '/dep000045/!d'))

echo "Processing ${#all_json_files[@]} files"


for json_file in "${all_json_files[@]}"; do
  if [[ ! -f "$json_file" ]]; then
    echo "No matching files found in ${json_directory}/"
    continue
  fi

  echo "Processing file: $json_file"

  deployment_id=$(basename "$json_file" | sed 's/_workload_chunks.json//')
  echo "Deployment ID: $deployment_id"

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
        ./slurm_scripts/singapore_sbatch.sh

  # if [ $? -ne 0 ]; then
  #   echo "Error submitting job for chunk $SLURM_ARRAY_TASK_ID of deployment $deployment_id"
  #   exit 1
  # fi

  # python 05_combine_outputs.py \
  #   --csv_file_pattern "${output_base_dir}/${deployment_id}/${deployment_id}_*.csv" \
  #   --main_csv_file "${output_base_dir}/${deployment_id}_cleaned.csv" \
  #   --remove_empty_rows


done
