#!/bin/bash

#SBATCH --job-name=nettlebed
#SBATCH --output=./logs/nettlebed.out
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --partition=orchid
#SBATCH --account=orchid

source ~/miniforge3/bin/activate
conda activate "~/conda_envs/flatbug/"

json_directory="./keys/nettlebed"
region="gbr"
output_base_dir="./data/nettlebed"
credentials_file="./credentials.json"

# amber-keys --bucket $region' --deployment_id 'dep000082' --output_file './keys/nettlebed/dep000082_keys_subset.json' --subset_dates '2024-08-01' '2024-08-02'


for json_file in ${json_directory}/dep*_keys_subset.json; do
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
#SBATCH --output=./logs/nettlebed/${deployment_id}_chunk_${chunk_id}.out
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
  --json_file "${json_file}" \
  --output_dir "${output_base_dir}" \
  --bucket_name $region \
  --credentials_file "$credentials_file" \
  --csv_file 'dep000082_${chunk_id}.csv' \
  --localisation_model_path ./models/fasterrcnn_resnet50_fpn_tz53qv9v.pt \
  --species_model_path ./models/turing-uk_v01_resnet50_2024-05-13-10-03_state.pt \
  --species_labels ./models/01_uk_data_category_map.json \
  --perform_inference \
  --save_crops
EOF

    if [ $? -ne 0 ]; then
      echo "Error submitting job for chunk $chunk_id of deployment $deployment_id"
      exit 1
    fi
  done
done


python 05_combine_outputs.py \
  --csv_file_pattern "./dep000082_*.csv" \
  --main_csv_file "./dep000082.csv"
