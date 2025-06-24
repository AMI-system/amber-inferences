#!/bin/bash
#SBATCH --time=10:00:00

echo "Job name: ${SLURM_JOB_NAME}"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Chunk ID: ${SLURM_ARRAY_TASK_ID}"


IFS=' ' read -r -a session_names_array <<< "$session_names_string"
this_session="${session_names_array[$SLURM_ARRAY_TASK_ID - 1]}"
echo "Session date: ${this_session}"

echo "Output csv: ${output_base_dir}/${deployment_id}/${deployment_id}_${this_session}.csv"


# Set up GPU monitoring log
GPU_LOG_DIR="${output_base_dir}/${deployment_id}/compute_resources"
mkdir -p "$GPU_LOG_DIR"
GPU_LOG_FILE="${GPU_LOG_DIR}/gpu_usage_task_${SLURM_ARRAY_TASK_ID}.csv"

# Start background GPU logger
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 10 > "$GPU_LOG_FILE" &
GPU_LOG_PID=$!


python3 -m amber_inferences.cli.perform_inferences \
  --chunk_id ${SLURM_ARRAY_TASK_ID} \
  --json_file "${json_file}" \
  --output_dir "${output_base_dir}" \
  --bucket_name "${region}" \
  --credentials_file "${credentials_file}" \
  --csv_file "${output_base_dir}/${deployment_id}/${deployment_id}_${this_session}.csv" \
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

python3 -m amber_inferences.cli.get_tracks \
  --csv_file "${output_base_dir}/${deployment_id}/${deployment_id}_${this_session}.csv" \
  --tracking_threshold 1

if [ $? -ne 0 ]; then
    echo "Error submitting job for chunk $SLURM_ARRAY_TASK_ID of deployment $deployment_id"
    exit 1
fi

sacct -j ${SLURM_JOBID} --format=JobID,MaxRSS,AveRSS,NNodes,Elapsed,TotalCPU,ReqMem,ReqTRES,TRESUsageOutMax,TRESUsageInMax >> "${output_base_dir}/${deployment_id}/compute_resources/sacct_${SLURM_ARRAY_TASK_ID}.txt"

trap "kill $GPU_LOG_PID" EXIT
