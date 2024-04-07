import contextlib
import numpy as np
import time
import torch
from typing import Dict
from habitat.core.logging import logger
from habitat.utils import profiling_wrapper
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensorboard_utils import (
    get_writer,
)

from habitat_baselines.rl.ddppo.ddp_utils import (
    EXIT,
    load_resume_state,
    rank0_only,
    requeue_job,
    save_resume_state,
)

from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from habitat_baselines.rl.ddppo.policy import PointNavResNetNet
from habitat_baselines.utils.common import (
    batch_obs,
    inference_mode
)
from habitat_baselines.common.obs_transformers import apply_obs_transforms_batch
from habitat_baselines.utils.info_dict import extract_scalars_from_infos
from habitat_baselines.utils.timing import g_timer

@baseline_registry.register_trainer(name="curriculum_warmup_zero_gps_trainer")
class CurriculumWarmupZeroGpsTrainer(PPOTrainer):
    
    def __init__(self, config=None):
        super().__init__(config)
        self.gps_available_every_x_steps = self.config.habitat.gps_available_every_x_steps
        self.has_found_human = 0
        self.warmup_steps = self.config.habitat.warmup_steps
        self.curriculum_upper_threshold = self.config.habitat.curriculum_upper_threshold
        self.curriculum_lower_threshold = self.config.habitat.curriculum_lower_threshold
        
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
                    
                    if self.num_steps_done >= self.warmup_steps:
                        # Adjust self.gps_available_every_x_steps based on the metric value
                        if self.has_found_human >= self.curriculum_upper_threshold:
                            # High success rate, decrease frequency of GPS updates
                            self.gps_available_every_x_steps = min(
                                self.gps_available_every_x_steps * 2, 
                                self.config.habitat.environment.max_episode_steps
                            )
                        elif self.has_found_human >= self.curriculum_lower_threshold:
                            # Moderate success rate, maintain the current frequency of GPS updates
                            pass  # No change needed
                        else:
                            # Low success rate, increase frequency of GPS updates
                            self.gps_available_every_x_steps = max(self.gps_available_every_x_steps // 2, 1)
                    # print("self.has_found_human", self.has_found_human)
                    # print("self.gps_available_every_x_steps", self.gps_available_every_x_steps)
                    # print("---")
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
                if step_id % self.gps_available_every_x_steps != 0:
                    with inference_mode():
                        batch['agent_0_goal_to_agent_gps_compass'][i] = torch.zeros_like(batch['agent_0_goal_to_agent_gps_compass'][i])
                # else:
                #     logger.info(batch['step_id'][i])
                #     logger.info(batch['agent_0_goal_to_agent_gps_compass'])
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