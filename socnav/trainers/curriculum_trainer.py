import contextlib
import os
import random
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Dict

import numpy as np
import torch
from omegaconf import OmegaConf

from habitat import logger
from habitat.config import read_write
from habitat.utils import profiling_wrapper
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import apply_obs_transforms_batch
from habitat_baselines.common.tensorboard_utils import get_writer
from habitat_baselines.rl.ddppo.ddp_utils import (
    EXIT,
    init_distrib_slurm,
    load_resume_state,
    rank0_only,
    requeue_job,
    save_resume_state,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

from habitat_baselines.rl.ddppo.policy import PointNavResNetNet
from habitat_baselines.utils.common import (
    batch_obs,
    inference_mode,
)
from habitat_baselines.utils.info_dict import (
    NON_SCALAR_METRICS,
    extract_scalars_from_infos,
)
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer, get_device
from habitat_baselines.rl.ddppo.policy import PointNavResNetNet
from habitat_baselines.utils.common import (
    batch_obs,
    inference_mode
)
from habitat_baselines.common.obs_transformers import apply_obs_transforms_batch
from habitat_baselines.utils.info_dict import extract_scalars_from_infos
from habitat_baselines.utils.timing import g_timer

@baseline_registry.register_trainer(name="curriculum_trainer")
class CurriculumTrainer(PPOTrainer):
    
    def __init__(self, config=None):
        super().__init__(config)
        self.gps_available_every_x_steps = self.config.habitat.gps_available_every_x_steps
        self.has_found_human = 0
        self.last_curr_update_step = 0

    def _init_train(self, resume_state=None):
        if resume_state is None:
            resume_state = load_resume_state(self.config)

        if resume_state is not None:
            if not self.config.habitat_baselines.load_resume_state_config:
                raise FileExistsError(
                    f"The configuration provided has habitat_baselines.load_resume_state_config=False but a previous training run exists. You can either delete the checkpoint folder {self.config.habitat_baselines.checkpoint_folder}, or change the configuration key habitat_baselines.checkpoint_folder in your new run."
                )

            self.config = self._get_resume_state_config_or_new_config(
                resume_state["config"]
            )
            self.gps_available_every_x_steps = resume_state["gps_available_every_x_steps"]

        if self.config.habitat_baselines.rl.ddppo.force_distributed:
            self._is_distributed = True

        self._add_preemption_signal_handlers()

        if self._is_distributed:
            local_rank, tcp_store = init_distrib_slurm(
                self.config.habitat_baselines.rl.ddppo.distrib_backend
            )
            if rank0_only():
                logger.info(
                    "Initialized DD-PPO with {} workers".format(
                        torch.distributed.get_world_size()
                    )
                )

            with read_write(self.config):
                self.config.habitat_baselines.torch_gpu_id = local_rank
                self.config.habitat.simulator.habitat_sim_v0.gpu_device_id = (
                    local_rank
                )
                # Multiply by the number of simulators to make sure they also get unique seeds
                self.config.habitat.seed += (
                    torch.distributed.get_rank()
                    * self.config.habitat_baselines.num_environments
                )

            random.seed(self.config.habitat.seed)
            np.random.seed(self.config.habitat.seed)
            torch.manual_seed(self.config.habitat.seed)
            self.num_rollouts_done_store = torch.distributed.PrefixStore(
                "rollout_tracker", tcp_store
            )
            self.num_rollouts_done_store.set("num_done", "0")

        if rank0_only() and self.config.habitat_baselines.verbose:
            logger.info(f"config: {OmegaConf.to_yaml(self.config)}")

        profiling_wrapper.configure(
            capture_start_step=self.config.habitat_baselines.profiling.capture_start_step,
            num_steps_to_capture=self.config.habitat_baselines.profiling.num_steps_to_capture,
        )

        # remove the non scalar measures from the measures since they can only be used in
        # evaluation
        for non_scalar_metric in NON_SCALAR_METRICS:
            non_scalar_metric_root = non_scalar_metric.split(".")[0]
            if non_scalar_metric_root in self.config.habitat.task.measurements:
                with read_write(self.config):
                    OmegaConf.set_struct(self.config, False)
                    self.config.habitat.task.measurements.pop(
                        non_scalar_metric_root
                    )
                    OmegaConf.set_struct(self.config, True)
                if self.config.habitat_baselines.verbose:
                    logger.info(
                        f"Removed metric {non_scalar_metric_root} from metrics since it cannot be used during training."
                    )

        self._init_envs()

        self.device = get_device(self.config)

        if rank0_only() and not os.path.isdir(
            self.config.habitat_baselines.checkpoint_folder
        ):
            os.makedirs(self.config.habitat_baselines.checkpoint_folder)

        logger.add_filehandler(self.config.habitat_baselines.log_file)

        self._agent = self._create_agent(resume_state)
        if self._is_distributed:
            self._agent.init_distributed(find_unused_params=False)  # type: ignore
        self._agent.post_init()

        self._is_static_encoder = (
            not self.config.habitat_baselines.rl.ddppo.train_encoder
        )
        self._ppo_cfg = self.config.habitat_baselines.rl.ppo

        observations = self.envs.reset()
        observations = self.envs.post_step(observations)
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        if self._is_static_encoder:
            self._encoder = self._agent.actor_critic.visual_encoder
            assert (
                self._encoder is not None
            ), "Visual encoder is not specified for this actor"
            with inference_mode():
                batch[
                    PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                ] = self._encoder(batch)

        ### NOTE: We initialize last GPS here based on the batch
        self.last_known_gps_obs = batch['agent_0_goal_to_agent_gps_compass'] #batch['step_id']
        ###
        self._agent.rollouts.insert_first_observations(batch)

        self.current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        self.running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        self.window_episode_stats = defaultdict(
            lambda: deque(maxlen=self._ppo_cfg.reward_window_size)
        )

        self.t_start = time.time()
        
    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        r"""Main method for training DD/PPO.

        Returns:
            None
        """

        resume_state = load_resume_state(self.config)
        self._init_train(resume_state)

        count_checkpoints = 0
        prev_time = 0

        if self._is_distributed:
            torch.distributed.barrier()

        resume_run_id = None
        if resume_state is not None:
            self._agent.load_state_dict(resume_state)

            requeue_stats = resume_state["requeue_stats"]
            self.num_steps_done = requeue_stats["num_steps_done"]
            self.num_updates_done = requeue_stats["num_updates_done"]
            self._last_checkpoint_percent = requeue_stats[
                "_last_checkpoint_percent"
            ]
            count_checkpoints = requeue_stats["count_checkpoints"]
            prev_time = requeue_stats["prev_time"]

            self.running_episode_stats = requeue_stats["running_episode_stats"]
            self.window_episode_stats.update(
                requeue_stats["window_episode_stats"]
            )
            resume_run_id = requeue_stats.get("run_id", None)

        with (
            get_writer(
                self.config,
                resume_run_id=resume_run_id,
                flush_secs=self.flush_secs,
                purge_step=int(self.num_steps_done),
            )
            if rank0_only()
            else contextlib.suppress()
        ) as writer:
            while not self.is_done():
                profiling_wrapper.on_start_step()
                profiling_wrapper.range_push("train update")

                self._agent.pre_rollout()

                if rank0_only() and self._should_save_resume_state():
                    requeue_stats = dict(
                        count_checkpoints=count_checkpoints,
                        num_steps_done=self.num_steps_done,
                        num_updates_done=self.num_updates_done,
                        _last_checkpoint_percent=self._last_checkpoint_percent,
                        prev_time=(time.time() - self.t_start) + prev_time,
                        running_episode_stats=self.running_episode_stats,
                        window_episode_stats=dict(self.window_episode_stats),
                        run_id=writer.get_run_id(),
                    )

                    save_resume_state(
                        dict(
                            **self._agent.get_resume_state(),
                            config=self.config,
                            requeue_stats=requeue_stats,
                            gps_available_every_x_steps=self.gps_available_every_x_steps
                        ),
                        self.config,
                    )

                if EXIT.is_set():
                    profiling_wrapper.range_pop()  # train update

                    self.envs.close()

                    requeue_job()

                    return

                self._agent.eval()
                count_steps_delta = 0
                profiling_wrapper.range_push("rollouts loop")

                profiling_wrapper.range_push("_collect_rollout_step")
                with g_timer.avg_time("trainer.rollout_collect"):
                    for buffer_index in range(self._agent.nbuffers):
                        self._compute_actions_and_step_envs(buffer_index)

                    for step in range(self._ppo_cfg.num_steps):
                        is_last_step = (
                            self.should_end_early(step + 1)
                            or (step + 1) == self._ppo_cfg.num_steps
                        )

                        for buffer_index in range(self._agent.nbuffers):
                            # NOTE: We pass step directly. 
                            # The step = 0, actually corresponds to the first step (we have already stepped once in the envs)
                            # TODO: Think if this is logically correct
                            count_steps_delta += (
                                self._collect_environment_result(buffer_index)
                            )

                            if (buffer_index + 1) == self._agent.nbuffers:
                                profiling_wrapper.range_pop()  # _collect_rollout_step

                            if not is_last_step:
                                if (buffer_index + 1) == self._agent.nbuffers:
                                    profiling_wrapper.range_push(
                                        "_collect_rollout_step"
                                    )

                                self._compute_actions_and_step_envs(
                                    buffer_index
                                )

                        if is_last_step:
                            break

                profiling_wrapper.range_pop()  # rollouts loop

                if self._is_distributed:
                    self.num_rollouts_done_store.add("num_done", 1)

                losses = self._update_agent()

                self.num_updates_done += 1
                losses = self._coalesce_post_step(
                    losses,
                    count_steps_delta,
                )
                ### Curriculum logic
                # NOTE: We implement the curriculum logic here
                # We get metrics here and decide on the next value
                # We compute metrics only on first GPU because that's how metrics are calculated for logging
                # Therefore, we only update the GPS availability according to those metrics
                if rank0_only():
                    deltas = {
                        k: (
                            (v[-1] - v[0]).sum().item()
                            if len(v) > 1
                            else v[0].sum().item()
                        )
                        for k, v in self.window_episode_stats.items()
                    }
                    deltas["count"] = max(deltas["count"], 1.0)


                    # Check to see if there are any metrics
                    # that haven't been logged yet
                    metrics = {
                        k: v / deltas["count"]
                        for k, v in deltas.items()
                        if k not in {"reward", "count"}
                    }
                    assert 'social_nav_stats.has_found_human' in metrics
                    self.has_found_human = metrics['social_nav_stats.has_found_human']
                    curriculum_config = self.config.habitat.curriculum_config
                    self.lower_threshold = curriculum_config.curriculum_lower_threshold
                    # Warmup
                    if self.num_steps_done >= curriculum_config.warmup_steps and self.num_steps_done >= self.last_curr_update_step + curriculum_config.update_curriculum_every_x_steps:
                        self.last_curr_update_step = self.num_steps_done
                        # Dynamic Additive Increment
                        if curriculum_config.dynamic_additive:
                            baseline = curriculum_config.dynamic_increment_baseline_score
                            increment_scale = curriculum_config.dynamic_increment_scaling_factor
                            dynamic_increment = int((self.has_found_human - baseline) // increment_scale)
                            decrement_scale = curriculum_config.dynamic_decrement_scaling_factor
                            dynamic_decrement = int((self.has_found_human - baseline) // decrement_scale)
                            
                        # Dynamic Threshold
                        if curriculum_config.use_dynamic_lower_threshold:
                            for idx, thresh in enumerate(curriculum_config.dynamic_lower_threshold):
                                if idx == 0:
                                    if self.num_steps_done <= thresh[0]:
                                        self.lower_threshold = thresh[1]
                                        break
                                else:
                                    if self.num_steps_done > curriculum_config.dynamic_lower_threshold[idx-1][0] and self.num_steps_done <= thresh[0]:
                                        self.lower_threshold = thresh[1]
                                        break
                        # Decrease GPS Freq
                        if self.has_found_human > curriculum_config.curriculum_upper_threshold:
                            if curriculum_config.additive:
                                self.gps_available_every_x_steps = min(
                                    self.gps_available_every_x_steps + curriculum_config.add_increment, 
                                    self.config.habitat.environment.max_episode_steps
                                )
                            elif self.config.habitat.curriculum_config.dynamic_additive:
                                self.gps_available_every_x_steps = min(
                                    self.gps_available_every_x_steps + dynamic_increment, 
                                    self.config.habitat.environment.max_episode_steps
                                )
                            else:
                                self.gps_available_every_x_steps = min(
                                    int(self.gps_available_every_x_steps * curriculum_config.mult_increment), 
                                    self.config.habitat.environment.max_episode_steps
                                )
                        
                        # Maintain GPS Freq
                        elif self.has_found_human > self.lower_threshold:
                            pass
                        
                        # Reduce GPS Freq
                        else:
                            if curriculum_config.additive:
                                self.gps_available_every_x_steps = max(
                                    self.gps_available_every_x_steps - curriculum_config.add_decrement,
                                    1
                                )
                            elif curriculum_config.dynamic_additive:
                                self.gps_available_every_x_steps = max(
                                    self.gps_available_every_x_steps + dynamic_decrement, 
                                    1
                                )
                            else:
                                self.gps_available_every_x_steps = max(
                                    self.gps_available_every_x_steps // curriculum_config.mult_decrement,
                                    1
                                )
                        
                    ###
                self._training_log(writer, losses, prev_time)

                # checkpoint model
                if rank0_only() and self.should_checkpoint():
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth",
                        dict(
                            step=self.num_steps_done,
                            wall_time=(time.time() - self.t_start) + prev_time,
                        ),
                    )
                    count_checkpoints += 1

                profiling_wrapper.range_pop()  # train update

            self.envs.close()
        
    
    def _collect_environment_result(self, buffer_index: int = 0):
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._agent.nbuffers),
            int((buffer_index + 1) * num_envs / self._agent.nbuffers),
        )

        with g_timer.avg_time("trainer.step_env"):
            outputs = [
                self.envs.wait_step_at(index_env)
                for index_env in range(env_slice.start, env_slice.stop)
            ]

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
        with g_timer.avg_time("trainer.update_stats"):
            observations = self.envs.post_step(observations)
            batch = batch_obs(observations, device=self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore
            for i in range(len(batch['step_id'])):
                step_id = batch['step_id'][i].item()
                if self.config.habitat.curriculum_config.last_gps:
                    if step_id == 1:
                        self.last_known_gps_obs[i] = batch['agent_0_goal_to_agent_gps_compass'][i]
                    if step_id % self.gps_available_every_x_steps != 0:
                        with inference_mode():
                            batch['agent_0_goal_to_agent_gps_compass'][i] = self.last_known_gps_obs[i]
                    else:
                        self.last_known_gps_obs[i] = batch['agent_0_goal_to_agent_gps_compass'][i]
                else:
                    if step_id % self.gps_available_every_x_steps != 0:
                        with inference_mode():
                            batch['agent_0_goal_to_agent_gps_compass'][i] = torch.zeros_like(batch['agent_0_goal_to_agent_gps_compass'][i])
            rewards = torch.tensor(
                rewards_l,
                dtype=torch.float,
                device=self.current_episode_reward.device,
            )
            rewards = rewards.unsqueeze(1)

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device=self.current_episode_reward.device,
            )
            done_masks = torch.logical_not(not_done_masks)

            self.current_episode_reward[env_slice] += rewards
            current_ep_reward = self.current_episode_reward[env_slice]
            self.running_episode_stats["reward"][env_slice] += current_ep_reward.where(done_masks, current_ep_reward.new_zeros(()))  # type: ignore
            self.running_episode_stats["count"][env_slice] += done_masks.float()  # type: ignore

            self._single_proc_infos = extract_scalars_from_infos(
                infos,
                ignore_keys=set(
                    k for k in infos[0].keys() if k not in self._rank0_keys
                ),
            )
            extracted_infos = extract_scalars_from_infos(
                infos, ignore_keys=self._rank0_keys
            )
            for k, v_k in extracted_infos.items():
                v = torch.tensor(
                    v_k,
                    dtype=torch.float,
                    device=self.current_episode_reward.device,
                ).unsqueeze(1)
                if k not in self.running_episode_stats:
                    self.running_episode_stats[k] = torch.zeros_like(
                        self.running_episode_stats["count"]
                    )
                self.running_episode_stats[k][env_slice] += v.where(done_masks, v.new_zeros(()))  # type: ignore

            self.current_episode_reward[env_slice].masked_fill_(
                done_masks, 0.0
            )

        if self._is_static_encoder:
            with inference_mode(), g_timer.avg_time("trainer.visual_features"):
                batch[
                    PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                ] = self._encoder(batch)

        self._agent.rollouts.insert(
            next_observations=batch,
            rewards=rewards,
            next_masks=not_done_masks,
            buffer_index=buffer_index,
        )

        self._agent.rollouts.advance_rollout(buffer_index)

        return env_slice.stop - env_slice.start

    @rank0_only
    def _training_log(
        self, writer, losses: Dict[str, float], prev_time: int = 0
    ):
        deltas = {
            k: (
                (v[-1] - v[0]).sum().item()
                if len(v) > 1
                else v[0].sum().item()
            )
            for k, v in self.window_episode_stats.items()
        }
        deltas["count"] = max(deltas["count"], 1.0)

        writer.add_scalar(
            "reward",
            deltas["reward"] / deltas["count"],
            self.num_steps_done,
        )

        # Check to see if there are any metrics
        # that haven't been logged yet
        metrics = {
            k: v / deltas["count"]
            for k, v in deltas.items()
            if k not in {"reward", "count"}
        }
        # NOTE: We also log this metric as this might help us know how the gps frequency changes
        writer.add_scalar("metrics/gps_available_every_x_steps", self.gps_available_every_x_steps, self.num_steps_done)
        writer.add_scalar("metrics/lower_threshold", self.lower_threshold, self.num_steps_done)
         
        for k, v in metrics.items():
            writer.add_scalar(f"metrics/{k}", v, self.num_steps_done)
        for k, v in losses.items():
            writer.add_scalar(f"learner/{k}", v, self.num_steps_done)

        for k, v in self._single_proc_infos.items():
            writer.add_scalar(k, np.mean(v), self.num_steps_done)

        fps = self.num_steps_done / ((time.time() - self.t_start) + prev_time)

        # Log perf metrics.
        writer.add_scalar("perf/fps", fps, self.num_steps_done)

        for timer_name, timer_val in g_timer.items():
            writer.add_scalar(
                f"perf/{timer_name}",
                timer_val.mean,
                self.num_steps_done,
            )

        # log stats
        if (
            self.num_updates_done % self.config.habitat_baselines.log_interval
            == 0
        ):
            logger.info(
                "update: {}\tfps: {:.3f}\t".format(
                    self.num_updates_done,
                    fps,
                )
            )

            logger.info(
                f"Num updates: {self.num_updates_done}\tNum frames {self.num_steps_done}"
            )

            logger.info(
                "Average window size: {}  {}".format(
                    len(self.window_episode_stats["count"]),
                    "  ".join(
                        "{}: {:.3f}".format(k, v / deltas["count"])
                        for k, v in deltas.items()
                        if k != "count"
                    ),
                )
            )
            perf_stats_str = " ".join(
                [f"{k}: {v.mean:.3f}" for k, v in g_timer.items()]
            )
            logger.info(f"\tPerf Stats: {perf_stats_str}")
            if self.config.habitat_baselines.should_log_single_proc_infos:
                for k, v in self._single_proc_infos.items():
                    logger.info(f" - {k}: {np.mean(v):.3f}")
