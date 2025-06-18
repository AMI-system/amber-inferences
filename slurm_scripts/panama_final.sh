#!/bin/bash

source ~/miniforge3/bin/activate
conda activate "~/conda_envs/flatbug/"

json_directory="./keys/panama_final"
region="pan"
output_base_dir="/gws/nopw/j04/ceh_generic/kgoldmann/panama_inferences"
credentials_file="./credentials.json"

mkdir -p "${output_base_dir}"
mkdir -p "${json_directory}"

# array of strings dep000017-22 and dep000083-92
dep_files=()
for i in {17..22}; do
  dep_files+=("dep$(printf '%06d' $i)")
done
for i in {83..92}; do
  dep_files+=("dep$(printf '%06d' $i)")
done

# create the key files, only needs to run once
for dep in "${dep_files[@]}"; do
  echo $dep
  amber-keys --bucket $region --deployment_id $dep --output_file "${json_directory}/${dep}.json"
done

# for each json file/deployment, create a slurm job
for json_file in ${json_directory}/dep*.json; do
  if [[ ! -f "$json_file" ]]; then
    echo "No matching files found in ${json_directory}/"
    continue
  fi

  echo "Processing file: $json_file"

  deployment_id=$(basename "$json_file" | sed 's/.json//')
  echo "Deployment ID: $deployment_id"
  mkdir -p "${output_base_dir}/${deployment_id}"

  session_names=()
  while IFS= read -r key; do
    session_names+=("$key")
  done < <(jq -r 'keys[]' $json_file)
  num_chunks=${#session_names[@]}

  echo "Number of sessions: $num_chunks"

  if [ -z "$num_chunks" ] || ! [[ "$num_chunks" =~ ^[0-9]+$ ]]; then
    echo "Error: Invalid number of chunks for $json_file"
    continue
  fi

  # Call the sbatch script for deployment using batches for arrays
  sbatch --job-name="${region}_${deployment_id}" \
    --gres gpu:1 \
    --partition orchid \
    --qos orchid \
    --account orchid \
    --mem 8G \
    --array=1-$num_chunks \
    --output="./logs/$region/${deployment_id}_batch_%a.out" \
  --export=ALL,\
json_file="$json_file",\
session_names_string="${session_names[*]}",\
output_base_dir="$output_base_dir",\
deployment_id="$deployment_id",\
region="$region",\
credentials_file="$credentials_file",\
species_model="./models/panama_moth-model_v01_resnet50_2023-01-24-09-51.pt",\
species_labels="./models/panama_moth-category-map_24Jan2023.json" \
  ./slurm_scripts/array_processor.sh

done
