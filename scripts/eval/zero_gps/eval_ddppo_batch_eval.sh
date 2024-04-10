#!/bin/bash

input_file="selected_checkpoints.txt"
experiment_name="socnav_ddppo_baseline_multi_gpu_full_sparse_100/seed_1"
ckpt_dir="data/socnav_checkpoints/${experiment_name}"


while read -r checkpoint_file_name; do
  # Extract checkpoint_id from the file name
  checkpoint_id=$(echo "$checkpoint_file_name" | cut -d. -f2)

  # Construct uuid, tensorboard_dir, and current_ckpt_dir
  uuid="ckpt_${checkpoint_id}"
  tensorboard_dir="tb/${experiment_name}/ind/${checkpoint_id}"
  current_ckpt_dir="${ckpt_dir}/${checkpoint_file_name}"

  echo "Ckpt id: $uuid - $checkpoint_id, ${tensorboard_dir}, ${current_ckpt_dir}, ${experiment_name}"

  sbatch --job-name="snav-${checkpoint_id}" \
    --output="slurm_logs/eval/snav-ind/${experiment_name}-${checkpoint_id}.out" \
    --error="slurm_logs/eval/snav-ind/${experiment_name}-${checkpoint_id}.err" \
    --gpus a40:1 \
    --nodes 1 \
    --cpus-per-task 10 \
    --qos short \
    --partition cvmlp-lab \
    --export "ALL,CHECKPOINT_DIR=${current_ckpt_dir},TENSORBOARD_DIR=${tensorboard_dir}" \
    scripts/eval/zero_gps/zero_gps_eval_single_ckpt.sh

done < "$input_file"
