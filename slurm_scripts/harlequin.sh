#!/bin/bash

#SBATCH --job-name=process_chunks
#SBATCH --output=./logs/harlequin_new_loc.out
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=long-serial

source ~/miniforge3/bin/activate
conda activate "~/conda_envs/moth_detector_env/"

json_directory="./keys/harlequin"
output_base_dir="./data/harlequin_new_loc_model/"
credentials_file="./credentials.json"


# get the keys for deployments of interest and chunk
cri_deps=('dep000034' 'dep000032' 'dep000038' 'dep000037' 'dep000033')
pan_deps=('dep000020' 'dep000021' 'dep000022' 'dep000083' 'dep000084' 'dep000086' 'dep000087' 'dep000088' 'dep000089' 'dep000090' 'dep000091' 'dep000092' 'dep000017' 'dep000018' )

# for dep in "${cri_deps[@]}"; do
#     echo $dep
#     region="cri"

#     python 02_generate_keys.py --bucket $region --deployment_id $dep --output_file "${json_directory}/${region}/${dep}_keys.txt"
#     python 03_pre_chop_files.py --input_file "${json_directory}/${region}/${dep}_keys.txt" --file_extensions "jpg" "jpeg" --chunk_size 50 --output_file "${json_directory}/${region}/${dep}_workload_chunks.json"
# done

# for dep in "${pan_deps[@]}"; do
#     echo $dep
#     region="pan"

#     python 02_generate_keys.py --bucket $region --deployment_id $dep --output_file "${json_directory}/${region}/${dep}_keys.txt"
#     python 03_pre_chop_files.py --input_file "${json_directory}/${region}/${dep}_keys.txt" --file_extensions "jpg" "jpeg" --chunk_size 50 --output_file "${json_directory}/${region}/${dep}_workload_chunks.json"
# done



# Perform the inferences
for json_file in ${json_directory}/*/dep00008*_workload_chunks.json; do

  if [[ ! -f "$json_file" ]]; then
    echo "No matching files found in ${json_directory}/"
    continue
  fi

  echo "Processing file: $json_file"

  region=$(basename "$(dirname -- "$json_file")")
  echo $region

  deployment_id=$(basename "$json_file" | sed 's/_workload_chunks.json//')
  echo "Region: $region, Deployment ID: $deployment_id"

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
#SBATCH --output=logs/harlequin_new_loc/${region}/${deployment_id}/chunk_${deployment_id}_${chunk_id}.out
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=short-serial

source ~/miniforge3/bin/activate
conda activate "~/conda_envs/moth_detector_env/"

echo $region

python -u 04_process_chunks.py \
  --chunk_id $chunk_id \
  --json_file "$json_file" \
  --output_dir "$output_base_dir/$deployment_id" \
  --bucket_name "$region" \
  --credentials_file "$credentials_file" \
  --csv_file "$output_base_dir/${deployment_id}.csv" \
  --localisation_model_path ./models/fasterrcnn_resnet50_fpn_tz53qv9v.pt \
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

echo "All chunk jobs submitted successfully."
