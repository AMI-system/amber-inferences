#!/bin/bash

#SBATCH --job-name=flatbug
#SBATCH --output=./logs/flatbug.out
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --partition=orchid
#SBATCH --account=orchid

source ~/miniforge3/bin/activate
conda activate "~/conda_envs/flatbug/"

json_directory="./keys/solar"
region="gbr"
output_base_dir="./data/flatbug"
credentials_file="./credentials.json"

for json_file in ${json_directory}/dep000072_workload_chunks.json; do
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
#SBATCH --output=./logs/flatbug/${deployment_id}_chunk_${chunk_id}.out
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=orchid
#SBATCH --account=orchid

source ~/miniforge3/bin/activate
conda activate "~/conda_envs/flatbug/"

python 04_process_chunks.py \
  --chunk_id $chunk_id \
  --json_file "$json_file" \
  --output_dir "$output_base_dir/$deployment_id" \
  --bucket_name "$region" \
  --credentials_file "$credentials_file" \
  --csv_file "$output_base_dir/${deployment_id}_${chunk_id}.csv" \
  --box_threshold 0.6 \
  --species_model_path ./models/turing-uk_v03_resnet50_2024-05-13-10-03_state.pt \
  --species_labels ./models/03_uk_data_category_map.json \
  --perform_inference \
  --save_crops \
  --remove_image
EOF

    if [ $? -ne 0 ]; then
      echo "Error submitting job for chunk $chunk_id of deployment $deployment_id"
      exit 1
    fi
  done
done
