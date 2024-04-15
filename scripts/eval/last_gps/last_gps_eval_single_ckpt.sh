#!/bin/bash
#SBATCH --job-name=s-eval-lg
#SBATCH --output=slurm_logs/eval/socialnav-eval-ddppo-lg-%j.out
#SBATCH --error=slurm_logs/eval/socialnav-eval-ddppo-lg-%j.err
#SBATCH --gpus a40:1
#SBATCH --cpus-per-task 10
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --signal=USR1@90
#SBATCH --requeue
#SBATCH --exclude=nestor,shakey
#SBATCH --partition=cvmlp-lab
#SBATCH --qos=short

export GLOG_minloglevel=2
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet

MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

source /srv/flash1/gchhablani3/miniforge3/etc/profile.d/conda.sh
conda deactivate
conda activate socnav


export PYTHONPATH=/srv/flash1/gchhablani3/spring_2024/socnav/habitat-sim/src_python:${PYTHONPATH}

DATA_PATH="data/datasets/hssd/rearrange"

srun python -um socnav.run \
    --config-name=experiments/ddppo_socnav_full_sparse.yaml \
    habitat.gps_available_every_x_steps=1500 \
    habitat_baselines.evaluate=True \
    habitat.dataset.split=val \
    habitat_baselines.video_dir="videos" \
    ++habitat_baselines.eval.video_option=[] \
    habitat_baselines.num_checkpoints=150 \
    habitat_baselines.total_num_steps=3e8 \
    habitat_baselines.num_environments=12 \
    habitat_baselines.checkpoint_folder=${CHECKPOINT_DIR} \
    habitat_baselines.eval_ckpt_path_dir=${CHECKPOINT_DIR} \
    habitat_baselines.tensorboard_dir=${TENSORBOARD_DIR} \
    habitat.task.actions.agent_0_base_velocity.longitudinal_lin_speed=10.0 \
    habitat.task.actions.agent_0_base_velocity.ang_speed=10.0 \
    habitat.task.actions.agent_0_base_velocity.allow_dyn_slide=True \
    habitat.task.actions.agent_0_base_velocity.enable_rotation_check_for_dyn_slide=False \
    habitat.task.actions.agent_1_oracle_nav_randcoord_action.human_stop_and_walk_to_robot_distance_threshold=-1.0 \
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
    habitat.task.end_on_success=False \
    habitat.task.slack_reward=-0.1 \
    habitat.environment.max_episode_steps=1500 \
    habitat.simulator.kinematic_mode=True \
    habitat.simulator.ac_freq_ratio=4 \
    habitat.simulator.ctrl_freq=120 \
    habitat.simulator.agents.agent_0.joint_start_noise=0.0 \
    habitat_baselines.load_resume_state_config=False \
    habitat_baselines.test_episode_count=500 \
    habitat_baselines.eval.extra_sim_sensors.third_rgb_sensor.height=1080 \
    habitat_baselines.eval.extra_sim_sensors.third_rgb_sensor.width=1920 \
    habitat.dataset.data_path=${DATA_PATH}/val/social_rearrange.json.gz \
    habitat_baselines.evaluator._target_="socnav.trainers.evaluators.last_gps_evaluator.LastGpsEvaluator" \
