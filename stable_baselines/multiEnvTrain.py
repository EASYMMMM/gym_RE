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
    parser = argparse.ArgumentParser("Train an RL agent using Stable Baselines3")
    parser.add_argument(
        "--algo",
        help="RL Algorithm (Soft Actor-Critic by default)",
        default="sac",
        type=str,
        required=False,
        choices=["sac", "td3", "ppo"],
    )
    parser.add_argument(
        "--env", type=str, default="HalfCheetahBulletEnv-v0", help="environment ID"
    )
    parser.add_argument(
        "-n",
        "--n-timesteps",
        help="Number of training timesteps",
        default=int(1e6),
        type=int,
    )
    parser.add_argument(
        "--save-freq",
        help="Save the model every n steps (if negative, no checkpoint)",
        default=-1,
        type=int,
    )
    parser.add_argument(
        "--model-name",
        help="Name of the model's save path",
        default="",
        type=str,
    )
    parser.add_argument(
        "--num-cpu",
        help="Number of processes to use",
        default=1,
        type=int,
    )
    args = parser.parse_args()

    env_id = args.env
    num_cpu = args.num_cpu
    n_timesteps = args.n_timesteps
    model_name = args.model_name + "_cpu" + str(num_cpu) + "_"
    # 存放在sb3model/文件夹下
    save_path = f"sb3model/{env_id}/{model_name}{args.algo}_{env_id}"

    # tensorboard log 路径
    tensorboard_log_path = f"tensorboard_log/{env_id}/{model_name}{args.algo}_{env_id}"
    tensorboard_log_name = f"{model_name}{args.algo}_{env_id}"

    # Instantiate and wrap the environment
    #env = gym.make(env_id)
    #env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    env = make_vec_env(env_id = env_id, n_envs = num_cpu, vec_env_cls=SubprocVecEnv)

    # Create the evaluation environment and callbacks
    eval_env = Monitor(gym.make(env_id))

    callbacks = [EvalCallback(eval_env, best_model_save_path=save_path)]

    # Save a checkpoint every n steps
    if args.save_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=args.save_freq, save_path=save_path, name_prefix="rl_model"
            )
        )

    algo = {
        "sac": SAC,
        "td3": TD3,
        "ppo": PPO,
    }[args.algo]

    n_actions = env.action_space.shape[0]

    # Tuned hyperparameters from https://github.com/DLR-RM/rl-baselines3-zoo
    hyperparams = {
        "sac": dict(
            batch_size=256,
            gamma=0.98,
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_starts=10000,
            buffer_size=int(5e5),
            tau=0.01,
            gradient_steps=-1,
        ),
        "td3": dict(
            batch_size=100,
            policy_kwargs=dict(net_arch=[400, 300]),
            learning_rate=1e-3,
            learning_starts=10000,
            buffer_size=int(1e6),
            train_freq=1,
            gradient_steps=1,
            action_noise=NormalActionNoise(
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
            ),
        ),
        "ppo": dict(
            batch_size=512,
            learning_rate=2.5e-4,
            gamma=0.99
        )
    }[args.algo]



    begin_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    model = algo("MlpPolicy", env, verbose=1, tensorboard_log = tensorboard_log_path, **hyperparams)
    try:
        model.learn(n_timesteps, callback=callbacks , tb_log_name = tensorboard_log_name )
    except KeyboardInterrupt:
        pass
    print('=====================================')
    print(f"Saving to {save_path}.zip")
    model.save(save_path)
    end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print('Started at: ' + begin_time)
    print('Ended at: ' + end_time)
    print('=====================================')
