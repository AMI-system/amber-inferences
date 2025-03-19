#!/bin/bash
#SBATCH --output="./logs/$region/%x_chunk2_%a.out"
#SBATCH --time=04:00:00


echo "Job name: ${SLURM_JOB_NAME}"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Chunk ID: ${SLURM_ARRAY_TASK_ID}"
echo "Output csv: ${output_base_dir}/${deployment_id}/${deployment_id}_${SLURM_ARRAY_TASK_ID}.csv"
echo "$json_file"
echo "$output_base_dir"
echo "$deployment_id"
echo "$region"
echo "$credentials_file"
echo $species_model
echo $species_labels

python 04_process_chunks.py \
    --chunk_id ${SLURM_ARRAY_TASK_ID} \
    --json_file "${json_file}" \
    --output_dir "${output_base_dir}" \
    --bucket_name "${region}" \
    --credentials_file "${credentials_file}" \
    --csv_file "${output_base_dir}/${deployment_id}/${deployment_id}_${SLURM_ARRAY_TASK_ID}.csv" \
    --species_model_path $species_model \ #./models/turing-costarica_v03_resnet50_2024-06-04-16-17_state.pt \
    --species_labels $species_labels \ #./models/03_costarica_data_category_map.json \
    --perform_inference \
    --remove_image \
    --box_threshold 0

if [ $? -ne 0 ]; then
    echo "Error submitting job for chunk $SLURM_ARRAY_TASK_ID of deployment $deployment_id"
    exit 1
fi
