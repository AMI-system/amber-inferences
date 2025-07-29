#!/bin/bash

source ~/miniforge3/bin/activate
conda activate "~/amber/"

json_directory="./keys/gb_eh_final"
region="gbr"
output_base_dir="./data/gbr_east_hendred_inferences"
credentials_file="./credentials.json"

mkdir -p "${output_base_dir}"
mkdir -p "${json_directory}"

# array of strings dep000064 only
dep_files=()
for i in 64; do
  dep_files+=("dep$(printf '%06d' $i)")
done

# create the key files, only needs to run once
# for dep in "${dep_files[@]}"; do
#  echo $dep
#  amber-keys --bucket $region --deployment_id $dep --output_file "${json_directory}/${dep}.json"
#  done

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
  sbatch --job-name="$deployment_id" \
    --partition standard \
    --account ceh_generic \
    --mem 8G \
    --qos long \
    --array=1-$number_of_batches \
    --output="./logs/$region/${deployment_id}_batch_%a.out" \
  --export=ALL,\
json_file="$json_file",\
output_base_dir="$output_base_dir",\
deployment_id="$deployment_id",\
region="$region",\
credentials_file="$credentials_file",\
species_model="./models/turing-uk_v03_resnet50_2024-05-13-10-03_state.pt",\
species_labels="./models/03_uk_data_category_map.json",\
batch_size=$batch_size \
  ./slurm_scripts/array_processor_no_flatbug.sh

done