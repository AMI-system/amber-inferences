#!/bin/bash

source ~/miniforge3/bin/activate
conda activate "~/conda_envs/flatbug/"

json_directory="./keys/thailand"
region="tha" # See bucket names here: https://s3-portal.jasmin.ac.uk/object-store/
output_base_dir="./data/thailand"
credentials_file="./credentials.json"
species_model="./models/turing-thailand_v01_resnet50_2024-11-21-16-28_state.pt"
species_labels="./models/02_thailand_data_category_map.json"

mkdir -p $json_directory
mkdir -p $output_base_dir

# step 1: get the deployments
python 01_print_deployments.py --subset_countries 'Thailand' > ${json_directory}/thailand_deployments.txt
deployments=($(grep -oP 'Deployment ID: \Kdep[0-9]+' ${json_directory}/thailand_deployments.txt))

echo "Thailand deployments: ${deployments[@]}"

for deployment_id in "${deployments[@]}"; do
  echo "Processing keys for deployment: $deployment_id"

  json_file="${json_directory}/${deployment_id}_workload_chunks.json"

  # for each deployment get the keys
  python 02_generate_keys.py \
    --bucket $region \
    --deployment_id $deployment_id \
    --output_file "${json_directory}/${deployment_id}_keys.txt"
  python 03_pre_chop_files.py \
    --input_file "${json_directory}/${deployment_id}_keys.txt" \
    --file_extensions 'jpg' 'jpeg' \
    --chunk_size 100 \
    --output_file $json_file

  if [ $? -ne 0 ]; then
    echo "Error getting keys for deployment $deployment_id"
    exit 1
  fi

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
        --export=ALL,json_file="$json_file",output_base_dir="$output_base_dir",deployment_id="$deployment_id",region="$region",credentials_file="$credentials_file",species_model="$species_model",species_labels="$species_labels" \
        ./slurm_scripts/slurm_script.sh

  if [ $? -ne 0 ]; then
    echo "Error submitting job for chunk $SLURM_ARRAY_TASK_ID of deployment $deployment_id"
    exit 1
  fi

done
