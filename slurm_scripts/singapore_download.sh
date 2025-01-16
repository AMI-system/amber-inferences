#!/bin/bash

#SBATCH --job-name=process_chunks
#SBATCH --output=./logs/singapore_download.out
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=long-serial

source ~/miniforge3/bin/activate
conda activate "~/conda_envs/moth_detector_env/"

json_directory="./keys/singapore/"
region="sgp"
output_base_dir="./data/singapore_download/"
credentials_file="./credentials.json"

# get the keys for deployments of interest and chunk
deps=('dep000045')

for dep in "${deps[@]}"; do
    echo $dep
    python 02_generate_keys.py --bucket $region --deployment_id $dep --output_file "${json_directory}/${dep}_keys.txt"
    python 03_pre_chop_files.py --input_file "${json_directory}/${dep}_keys.txt" --file_extensions "jpg" "jpeg" --chunk_size 100 --output_file "${json_directory}/${dep}_workload_chunks.json"
done

for json_file in ${json_directory}/*_workload_chunks.json; do
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
#SBATCH --output=logs/singapore/${deployment_id}/chunk_${deployment_id}_${chunk_id}.out
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=short-serial

source ~/miniforge3/bin/activate
conda activate "~/conda_envs/moth_detector_env/"

python 04_process_chunks.py \
  --chunk_id $chunk_id \
  --json_file "$json_file" \
  --output_dir "$output_base_dir/$deployment_id" \
  --bucket_name "$region" \
  --credentials_file "$credentials_file"
EOF

    if [ $? -ne 0 ]; then
      echo "Error submitting job for chunk $chunk_id of deployment $deployment_id"
      exit 1
    fi
  done
done

echo "All chunk jobs submitted successfully."
