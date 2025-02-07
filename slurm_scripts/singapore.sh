#!/bin/bash

#SBATCH --job-name=singapore
#SBATCH --output=./logs/singapore.out
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --partition=orchid
#SBATCH --account=orchid

source ~/miniforge3/bin/activate
conda activate "~/conda_envs/flatbug/"

json_directory="./keys/singapore"
region="gbr"
output_base_dir="./data/singapore"
credentials_file="./credentials.json"

# array of strings dep000045 to dep000054
dep_files=()
for i in {45..54}; do
  dep_files+=("dep$(printf '%06d' $i)")
done

# create the key files
# for dep in "${dep_files[@]}"; do
#   echo $dep
#   amber-keys --bucket 'sgp' --deployment_id $dep --output_file "./keys/singapore/${dep}_workload_chunks.json"
# done

for json_file in ${json_directory}/dep*_workload_chunks.json; do
  if [[ ! -f "$json_file" ]]; then
    echo "No matching files found in ${json_directory}/"
    continue
  fi

  echo "Processing file: $json_file"

  deployment_id=$(basename "$json_file" | sed 's/_workload_chunks.json//')
  echo "Deployment ID: $deployment_id"

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

  for chunk_id in $(seq 1 2); do #"$num_chunks"); do
    echo "Submitting job for chunk $chunk_id of deployment $deployment_id"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=chunk_${deployment_id}_${chunk_id}
#SBATCH --output=./logs/singapore/${deployment_id}_chunk_${chunk_id}.out
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=orchid
#SBATCH --account=orchid

source ~/miniforge3/bin/activate
conda activate "~/conda_envs/flatbug/"

python3 amber_inferences.cli.perform_inferences \
  --chunk_id 1 \
  --json_file "${json_directory}/${deployment_id}_workload_chunks.json" \
  --output_dir './data/singapore' \
  --bucket_name 'sgp' \
  --credentials_file './credentials.json' \
  --csv_file '${deployment_id}_${chunk_id}.csv' \
  --localisation_model_path ./models/flat_bug_M.pt \
  --device 'cuda:0' \
  --species_model_path ./models/turing-singapore_v02_resnet50_2024-11-21-19-58_state.pt \
  --species_labels ./models/02_singapore_data_category_map.json \
  --perform_inference \
  --remove_image \
  --save_crops
EOF

    if [ $? -ne 0 ]; then
      echo "Error submitting job for chunk $chunk_id of deployment $deployment_id"
      exit 1
    fi
  done
done
