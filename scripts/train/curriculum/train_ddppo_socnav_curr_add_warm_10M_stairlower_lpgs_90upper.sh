#!/bin/bash
#SBATCH --job-name=snav-curr-add-warm-10M-stair-lg
#SBATCH --output=slurm_logs/train/snav-curr-add-warm-10M-stair-lg-%j.out
#SBATCH --error=slurm_logs/train/snav-curr-add-warm-10M-stair-lg-%j.err
#SBATCH --gpus a40:4
#SBATCH --cpus-per-task 10
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 4
#SBATCH --signal=USR1@90
#SBATCH --requeue
#SBATCH --exclude=shakey,chappie,kitt
#SBATCH --partition=cvmlp-lab
#SBATCH --qos=short

export GLOG_minloglevel=2
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet

MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

source /srv/flash1/skailas3/miniforge3/etc/profile.d/conda.sh
conda deactivate
conda activate socnav

# TENSORBOARD_DIR="tb/ddppo_socnav/seed_1/"
# CHECKPOINT_DIR="data/new_checkpoints/ddppo_socnav/seed_1/"

# export HABITAT_ENV_DEBUG=1
# export HYDRA_FULL_ERROR=1
export PYTHONPATH=/srv/flash1/gchhablani3/spring_2024/socnav/habitat-sim/src_python:${PYTHONPATH}

# wandb config
JOB_ID="socnav_ddppo_full_curriculum_additive_lower_staircase_lastgps_90_upper_warm_10M"
# split="train"
DATA_PATH="data/datasets/hssd/rearrange"
WB_ENTITY="sivakailas"
PROJECT_NAME="socnav"

TENSORBOARD_DIR="tb/${JOB_ID}/seed_4/"
CHECKPOINT_DIR="data/socnav_checkpoints/${JOB_ID}/seed_4/"


srun python -um socnav.run \
    --config-name=experiments/ddppo_socnav_full_curriculum.yaml \
    habitat.gps_available_every_x_steps=1 \
    habitat.curriculum_config.update_curriculum_every_x_steps=1000000 \
    habitat.curriculum_config.additive=True \
    habitat.curriculum_config.warmup_steps=10000000 \
    habitat.curriculum_config.curriculum_upper_threshold=0.9 \
    habitat.curriculum_config.last_gps=True \
    habitat.curriculum_config.use_dynamic_lower_threshold=True \
    habitat_baselines.evaluate=False \
    habitat_baselines.wb.entity=$WB_ENTITY \
    habitat_baselines.wb.run_name=$JOB_ID \
    habitat_baselines.wb.project_name=$PROJECT_NAME \
    habitat_baselines.num_checkpoints=150 \
    habitat_baselines.total_num_steps=3e8 \
    habitat_baselines.num_environments=32 \
    habitat_baselines.tensorboard_dir=${TENSORBOARD_DIR} \
    habitat_baselines.video_dir="videos" \
    habitat_baselines.load_resume_state_config=True \
    habitat_baselines.checkpoint_folder=${CHECKPOINT_DIR} \
    habitat_baselines.eval_ckpt_path_dir=${CHECKPOINT_DIR} \
    habitat.task.actions.agent_0_base_velocity.longitudinal_lin_speed=10.0 \
    habitat.task.actions.agent_0_base_velocity.ang_speed=10.0 \
    habitat.task.actions.agent_0_base_velocity.allow_dyn_slide=True \
    habitat.task.actions.agent_0_base_velocity.enable_rotation_check_for_dyn_slide=False \
    habitat.task.actions.agent_1_oracle_nav_randcoord_action.lin_speed=10.0 \
    habitat.task.actions.agent_1_oracle_nav_randcoord_action.ang_speed=10.0 \
    habitat.task.actions.agent_1_oracle_nav_action.lin_speed=10.0 \
    habitat.task.actions.agent_1_oracle_nav_action.ang_speed=10.0 \
    habitat.task.measurements.social_nav_reward.facing_human_reward=3.0 \
    habitat.task.measurements.social_nav_reward.count_coll_pen=0.01 \
    habitat.task.measurements.social_nav_reward.max_count_colls=-1 \
    habitat.task.measurements.social_nav_reward.count_coll_end_pen=5 \
    habitat.task.measurements.social_nav_reward.use_geo_distance=True \
    habitat.task.measurements.social_nav_reward.facing_human_dis=3.0 \
    habitat.task.measurements.social_nav_seek_success.following_step_succ_threshold=400 \
    habitat.task.measurements.social_nav_seek_success.need_to_face_human=True \
    habitat.task.measurements.social_nav_seek_success.use_geo_distance=True \
    habitat.task.measurements.social_nav_seek_success.facing_threshold=0.5 \
    habitat.task.lab_sensors.humanoid_detector_sensor.return_image=True \
    habitat.task.lab_sensors.humanoid_detector_sensor.is_return_image_bbox=True \
    habitat.task.success_reward=10.0 \
    habitat.task.end_on_success=True \
    habitat.task.slack_reward=-0.1 \
    habitat.environment.max_episode_steps=1500 \
    habitat.simulator.kinematic_mode=True \
    habitat.simulator.ac_freq_ratio=4 \
    habitat.simulator.ctrl_freq=120 \
    habitat.simulator.agents.agent_0.joint_start_noise=0.0 \
    habitat.dataset.data_path=${DATA_PATH}/train/social_rearrange.json.gz \
    # habitat.dataset.content_scenes=['102817140'] \
