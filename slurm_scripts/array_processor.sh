#!/bin/bash
#SBATCH --time=04:00:00

echo "Job name: ${SLURM_JOB_NAME}"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Chunk ID: ${SLURM_ARRAY_TASK_ID}"
echo "Output csv: ${output_base_dir}/${deployment_id}/${deployment_id}_${SLURM_ARRAY_TASK_ID}.csv"

batch_number_padded=$(printf "%04d" $SLURM_ARRAY_TASK_ID)

python3 -m amber_inferences.cli.perform_inferences \
  --chunk_id ${SLURM_ARRAY_TASK_ID} \
  --batch_size $batch_size \
  --json_file "${json_file}" \
  --output_dir "${output_base_dir}" \
  --bucket_name "${region}" \
  --credentials_file "${credentials_file}" \
  --csv_file "${output_base_dir}/${deployment_id}/${deployment_id}_${batch_number_padded}.csv" \
  --species_model_path $species_model \
  --species_labels $species_labels \
  --perform_inference \
  --remove_image \
  --box_threshold 0 \
  --binary_model_path ./models/moth-nonmoth-effv2b3_20220506_061527_30.pth \
  --localisation_model_path ./models/flat_bug_M.pt \
  --order_model_path ./models/dhc_best_128.pth \
  --order_thresholds_path ./models/thresholdsTestTrain.csv \
  --skip_processed

if [ $? -ne 0 ]; then
    echo "Error submitting job for chunk $SLURM_ARRAY_TASK_ID of deployment $deployment_id"
    exit 1
fi
