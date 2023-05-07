# Code adapted from https://github.com/DLR-RM/rl-baselines3-zoo
# it requires stable-baselines3 to be installed
# Colab Notebook: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/pybullet.ipynb
# You can run it using: python -m pybullet_envs.stable_baselines.train --algo td3 --env HalfCheetahBulletEnv-v0
# Author: Antonin RAFFIN
# MIT License

'''
开启多个环境同时训练
python stable_baselines/multiEnvTrain.py --algo td3 --env HalfCheetah-v2

python stable_baselines/multiEnvTrain.py --algo sac --env Humanoid-v3  --model-name 2e6

python stable_baselines/multiEnvTrain.py --algo ppo --env Humanoid-v3  --n-timesteps 2000000 --model-name 2e6

python stable_baselines/multiEnvTrain.py --algo ppo --env HumanoidCustomEnv-v0 --num-cpu 2 --n-timesteps 2000000 --model-name 2e6_t4 

python stable_baselines/multiEnvTrain.py --algo sac --env HumanoidCustomEnv-v0 --num-cpu 1 --n-timesteps 2000000 --model-name 2e6_t4 

python stable_baselines/multiEnvTrain.py --algo sac --env HumanoidCustomEnv-v0 --num-cpu 8 --n-timesteps 2000000 --model-name 2e6_ladder_t1 --terrain-type ladders 

'''
import argparse

# ------- 来自于mujoco150在win+py3.9下的矫情的要求 --------
# 手动添加mujoco路径
import os
from getpass import getuser
user_id = getuser()
os.add_dll_directory(f"C://Users//{user_id}//.mujoco//mujoco200//bin")
os.add_dll_directory(f"C://Users//{user_id}//.mujoco//mujoco-py-2.0.2.0//mujoco_py")
# -------------------------------------------------------
import pybullet_envs  # register pybullet envs
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import gym_custom_env       # 注册自定义环境
import time
import gym
import torch
import numpy as np
from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from rewardChecker import update_info_buffer,dump_logs


def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


if __name__ == "__main__":
  

    algo = 'sac'
    env_id = 'HumanoidCustomEnv-v0'
    num_cpu = 10
    n_timesteps = 2500000
    terrain_type = 'steps'
    if algo == 'sac':
        BaseAlgorithm._update_info_buffer = update_info_buffer
        OffPolicyAlgorithm._dump_logs = dump_logs

    
    # env kwargs
    env_kwargs = {'terrain_type':terrain_type}

# ========================= 原版 ======================================
    # 存放在sb3model/文件夹下
    save_path = f"sb3model/steps_exp/normal"

    # tensorboard log 路径
    tensorboard_log_path = f"tensorboard_log/steps_experiments/"
    tensorboard_log_name = f"normal_steps_no_evo"
        # env kwargs
    env_kwargs = {'terrain_type':terrain_type}

    env = make_vec_env(env_id = env_id, n_envs = num_cpu, env_kwargs = env_kwargs)

    # Create the evaluation environment and callbacks
    eval_env = Monitor(gym.make(env_id,terrain_type = terrain_type))

    callbacks = [EvalCallback(eval_env, best_model_save_path=save_path)]

    n_actions = env.action_space.shape[0]

    # Tuned hyperparameters from https://github.com/DLR-RM/rl-baselines3-zoo
    hyperparams = dict(
            batch_size=256,
            gamma=0.98,
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_starts=10000,
            buffer_size=int(5e5),
            tau=0.01,
            gradient_steps=4,
        )


    begin_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log = tensorboard_log_path, **hyperparams)
    model.learn(n_timesteps, callback=callbacks , tb_log_name = tensorboard_log_name )

    print('=====================================')
    print(f"Saving to {save_path}.zip")
    model.save(save_path)
    end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print('Started at: ' + begin_time)
    print('Ended at: ' + end_time)
    print('=====================================')

    del env
    del model
    del eval_env
    del callbacks

# ========================= No posture ======================================
    # 存放在sb3model/文件夹下
    save_path = f"sb3model/steps_exp/steps_no_poseture_R"

    # tensorboard log 路径
    tensorboard_log_path = f"tensorboard_log/steps_experiments/"
    tensorboard_log_name = f"steps_no_poseture_R"

    env_kwargs = {'terrain_type':terrain_type, 'posture_reward_weight':0}

    env = make_vec_env(env_id = env_id, n_envs = num_cpu, env_kwargs = env_kwargs)

    # Create the evaluation environment and callbacks
    eval_env = Monitor(gym.make(env_id,terrain_type = terrain_type))

    callbacks = [EvalCallback(eval_env, best_model_save_path=save_path)]

    n_actions = env.action_space.shape[0]

    # Tuned hyperparameters from https://github.com/DLR-RM/rl-baselines3-zoo
    hyperparams = dict(
            batch_size=256,
            gamma=0.98,
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_starts=10000,
            buffer_size=int(5e5),
            tau=0.01,
            gradient_steps=4,
        )


    begin_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log = tensorboard_log_path, **hyperparams)
    model.learn(n_timesteps, callback=callbacks , tb_log_name = tensorboard_log_name )

    print('=====================================')
    print(f"Saving to {save_path}.zip")
    model.save(save_path)
    end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print('Started at: ' + begin_time)
    print('Ended at: ' + end_time)
    print('=====================================')

    del env
    del model
    del eval_env
    del callbacks


# ========================= No terrain_obs ======================================
    # 存放在sb3model/文件夹下
    save_path = f"sb3model/steps_exp/steps_no_terrain_info"

    # tensorboard log 路径
    tensorboard_log_path = f"tensorboard_log/steps_experiments/"
    tensorboard_log_name = f"steps_no_terrain_info"

    env_kwargs = {'terrain_type':terrain_type, 'terrain_info':False}

    env = make_vec_env(env_id = env_id, n_envs = num_cpu, env_kwargs = env_kwargs)

    # Create the evaluation environment and callbacks
    eval_env = Monitor(gym.make(env_id,terrain_type = terrain_type))

    callbacks = [EvalCallback(eval_env, best_model_save_path=save_path)]

    n_actions = env.action_space.shape[0]

    # Tuned hyperparameters from https://github.com/DLR-RM/rl-baselines3-zoo
    hyperparams = dict(
            batch_size=256,
            gamma=0.98,
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_starts=10000,
            buffer_size=int(5e5),
            tau=0.01,
            gradient_steps=4,
        )


    begin_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log = tensorboard_log_path, **hyperparams)
    model.learn(n_timesteps, callback=callbacks , tb_log_name = tensorboard_log_name )

    print('=====================================')
    print(f"Saving to {save_path}.zip")
    model.save(save_path)
    end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print('Started at: ' + begin_time)
    print('Ended at: ' + end_time)
    print('=====================================')

    del env
    del model
    del eval_env
    del callbacks
