'''
重新定义内部函数，将训练中的info数据打印出来
'''
import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import time,sys
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps


def update_info_buffer(self, infos: List[Dict[str, Any]], dones: Optional[np.ndarray] = None) -> None:
    """
    对BaseAlgorithm中_update_info_buffer方法的覆写。
    在episode_buffer中添加奖励函数细节

    Retrieve reward, episode length, episode success and update the buffer
    if using Monitor wrapper or a GoalEnv.

    :param infos: List of additional information about the transition.
    :param dones: Termination signals
    """
    if dones is None:
        dones = np.array([False] * len(infos))
    for idx, info in enumerate(infos):
        maybe_ep_info = info.get("episode") # info.["episode"]为字典，在monitor.py中添加在env的info中。
        maybe_is_success = info.get("is_success")
        if maybe_ep_info is not None:
            maybe_ep_info.update(info.get("reward_details"))
            maybe_ep_info.update(info.get("xyz_position"))
            self.ep_info_buffer.extend([maybe_ep_info])
        if maybe_is_success is not None and dones[idx]:
            self.ep_success_buffer.append(maybe_is_success)


def dump_logs(self) -> None:
    """
    对off_policy_algorithm中_dump_logs方法的覆写。
    在logger中添加奖励函数的细节
    Write log.
    """
    time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
    fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
    self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
    if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
        self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
        self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("details/ep_forward_rew_mean", safe_mean([ep_info["forward_reward_sum"] for ep_info in self.ep_info_buffer]), exclude=("stdout", "log"))
        self.logger.record("details/ep_posture_rew_mean", safe_mean([ep_info["posture_reward_sum"] for ep_info in self.ep_info_buffer]), exclude=("stdout", "log"))
        self.logger.record("details/ep_contact_rew_mean", safe_mean([ep_info["contact_reward_sum"] for ep_info in self.ep_info_buffer]), exclude=("stdout", "log"))
        self.logger.record("details/ep_healthy_rew_mean", safe_mean([ep_info["healthy_reward_sum"] for ep_info in self.ep_info_buffer]), exclude=("stdout", "log"))
        self.logger.record("details/ep_contact_cost_mean", safe_mean([ep_info["contact_cost_sum"] for ep_info in self.ep_info_buffer]), exclude=("stdout", "log"))
        self.logger.record("details/ep_control_cost_mean", safe_mean([ep_info["control_cost_sum"] for ep_info in self.ep_info_buffer]), exclude=("stdout", "log"))
        self.logger.record("details/final_x", safe_mean([ep_info["final_x"] for ep_info in self.ep_info_buffer]), exclude=("stdout", "log"))
    self.logger.record("time/fps", fps)
    self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
    self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
    if self.use_sde:
        self.logger.record("train/std", (self.actor.get_std()).mean().item())

    if len(self.ep_success_buffer) > 0:
        self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
    # Pass the number of timesteps for tensorboard
    self.logger.dump(step=self.num_timesteps)

