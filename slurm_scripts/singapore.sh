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
region="sgp"
output_base_dir="./data/singapore"
credentials_file="./credentials.json"

# amber-keys --bucket $region' --deployment_id 'dep000082' --output_file './keys/singapore/dep000082_keys_subset.json' --subset_dates '2024-08-01' '2024-08-02'

# get all json files in the directory
all_json_files=(${json_directory}/dep*_workload_chunks.json)

# remove the ones containing dep000045, 46, 51 (already run)
all_json_files=(${all_json_files[@]//*dep000045*/})
all_json_files=(${all_json_files[@]//*dep000046*/})
all_json_files=(${all_json_files[@]//*dep000051*/})

echo "Processing ${#all_json_files[@]} files"


for json_file in "${all_json_files[@]}"; do
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

  for chunk_id in $(seq 1 "$num_chunks"); do
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

mkdir ${output_base_dir}/${deployment_id}

python 04_process_chunks.py \
  --chunk_id $chunk_id \
  --json_file "${json_file}" \
  --output_dir "${output_base_dir}" \
  --bucket_name $region \
  --credentials_file "$credentials_file" \
  --csv_file '${output_base_dir}/${deployment_id}/${deployment_id}_${chunk_id}.csv' \
  --species_model_path ./models/turing-singapore_v02_resnet50_2024-11-21-19-58_state.pt \
  --species_labels ./models/02_singapore_data_category_map.json \
  --perform_inference \
  --remove_image \
  --box_threshold 0.01
EOF

    if [ $? -ne 0 ]; then
      echo "Error submitting job for chunk $chunk_id of deployment $deployment_id"
      exit 1
    fi
  done
done
