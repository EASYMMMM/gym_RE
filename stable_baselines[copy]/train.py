# Code adapted from https://github.com/DLR-RM/rl-baselines3-zoo
# it requires stable-baselines3 to be installed
# Colab Notebook: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/pybullet.ipynb
# You can run it using: python -m pybullet_envs.stable_baselines.train --algo td3 --env HalfCheetahBulletEnv-v0
# Author: Antonin RAFFIN
# MIT License

'''
来自pybullet的sb3例程
作出部分更改
python stable_baselines[copy]/train.py --algo td3 --env HalfCheetah-v2

python stable_baselines[copy]/train.py --algo sac --env Humanoid-v3  --model-name 2e6

python stable_baselines[copy]/train.py --algo ppo --env Humanoid-v3  --n-timesteps 2e6 --model-name 2e6 
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

import gym
import numpy as np
from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor


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
    args = parser.parse_args()

    env_id = args.env
    n_timesteps = args.n_timesteps
    model_name = args.model_name + "_"
     # 存放在sb3model/文件夹下
    save_path = f"sb3model/{model_name}{args.algo}_{env_id}"

    # Instantiate and wrap the environment
    env = gym.make(env_id)

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
            buffer_size=int(3e5),
            tau=0.01,
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

    model = algo("MlpPolicy", env, verbose=1, **hyperparams)
    try:
        model.learn(n_timesteps, callback=callbacks)
    except KeyboardInterrupt:
        pass

    print(f"Saving to {save_path}.zip")
    model.save(save_path)
