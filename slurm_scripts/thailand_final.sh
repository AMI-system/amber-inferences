#!/bin/bash

source ~/miniforge3/bin/activate
conda activate "~/conda_envs/flatbug/"

json_directory="./keys/thailand_final"
region="tha"
output_base_dir="/gws/nopw/j04/ceh_generic/kgoldmann/thailand_inferences_tracking"
credentials_file="./credentials.json"

mkdir -p "${output_base_dir}"
mkdir -p "${json_directory}"

# array of strings dep000074 to dep000078
dep_files=()
for i in {74..78}; do
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

  num_chunks=$(wc -l $json_file | awk '{print $1-2}')

  echo "Number of chunks: $num_chunks"

  if [ -z "$num_chunks" ] || ! [[ "$num_chunks" =~ ^[0-9]+$ ]]; then
    echo "Error: Invalid number of chunks for $json_file"
    continue
  fi

  # note: slurm will not accept array indices > 10,000 so we need to break
  # the array down into manageable chunks
  batch_size=1000
  number_of_batches=$((num_chunks / batch_size + 1))

  # Call the sbatch script for deployment using batches for arrays
  sbatch --job-name="${region}_${deployment_id}" \
    --gres gpu:1 \
    --partition orchid \
    --qos orchid \
    --account orchid \
    --mem 8G \
    --array=1-$number_of_batches \
    --output="./logs/$region/${deployment_id}_batch_%a.out" \
  --export=ALL,\
json_file="$json_file",\
output_base_dir="$output_base_dir",\
deployment_id="$deployment_id",\
region="$region",\
credentials_file="$credentials_file",\
species_model="./models/turing-thailand_v01_resnet50_2024-11-21-16-28_state.pt",\
species_labels="./models/02_thailand_data_category_map.json",\
batch_size=$batch_size \
  ./slurm_scripts/array_processor.sh

done
