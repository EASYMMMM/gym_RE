'''
自定义的回调函数
'''
import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union
import time
import gym
import numpy as np

try:
    from tqdm import TqdmExperimentalWarning
    # Remove experimental warning
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
    from tqdm.rich import tqdm
except ImportError:
    # Rich not installed, we only throw an error
    # if the progress bar is used
    tqdm = None

from stable_baselines3.common import base_class  # pytype: disable=pyi-error
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from stable_baselines3.common.callbacks import EventCallback, EvalCallback , BaseCallback

from GA import GA_Design_Optim
from gym_custom_env import HumanoidXML

class EvolutionCallback(EventCallback):
    """
    在EvalCallback的基础上更改
    Callback for evaluating an agent.

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        total_steps: int ,
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        warm_up_steps: int = 1000000,
        design_update_steps: int = 200000,
        overchange_punish: int = 0,
        terrain_type = 'steps',
        pop_size:int = 50, #种群大小
    ):
        super().__init__(callback_after_eval, verbose=verbose)

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn
        self.total_steps = total_steps

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []

        self.warm_up_steps = warm_up_steps
        self.design_update_steps = design_update_steps
        

        self.overchange_punish = overchange_punish
        self.terrain_type = terrain_type
        self.pop_size = pop_size

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

    def _on_training_start(self) -> None:
        # GA 优化器
        self.GA_design_optimizer = GA_Design_Optim(self.model,decode_size = 20,
                                                     POP_size = self.pop_size, n_generations = 5, overchange_punish= self.overchange_punish,
                                                     terrain_type= self.terrain_type  )
        self.last_time_trigger = self.num_timesteps
    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:
        # 每隔一定训练步长，进行形态优化

        continue_training = True

        if (self.num_timesteps - self.last_time_trigger) >= self.design_update_steps:
            if self.model._total_timesteps - self.num_timesteps < self.warm_up_steps :
                # 在结束时，进行收尾训练
                return continue_training
            # 每隔一定步数，进行设计参数更新
            self.last_time_trigger = self.num_timesteps
            new_design_params = self.GA_design_optimizer.evolve()
            self.model.env.update_xml_model(new_design_params)
            self.logger.record('design/thigh_lenth',new_design_params['thigh_lenth'])
            self.logger.record('design/shin_lenth',new_design_params['shin_lenth'])
            self.logger.record('design/upper_arm_lenth',new_design_params['upper_arm_lenth'])
            self.logger.record('design/lower_arm_lenth',new_design_params['lower_arm_lenth'])
            self.logger.record('design/foot_lenth',new_design_params['foot_lenth'])
            self.logger.record('parameter/overchange_punish',self.overchange_punish)
            self.last_params = new_design_params

        return continue_training

    def _on_training_end(self) -> None:
        xml_model = HumanoidXML(terrain_type=self.terrain_type)
        xml_model.set_params(self.last_params)
        end_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        xml_model.update_xml(file_path="gym_custom_env/assets/"+end_time+"humanoid_optim_result")


    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)

