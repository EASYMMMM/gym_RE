'''
python RL_train/pendulum_train.py 
'''
import argparse

# ------- 来自于mujoco150在win+py3.9下的矫情的要求 --------
# 手动添加mujoco路径
import os
from getpass import getuser
user_id = getuser()
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import gym_env       # 注册自定义环境
import time
import gym
import torch
import numpy as np
from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env


def main():
    parser = argparse.ArgumentParser("Train an RL agent using Stable Baselines3")

    # 随机种子
    seed = 1

    # 环境名
    env_id = 'InvertedPendulumEnv-v0'
    n_timesteps = 2000000
    model_name = 'InvPend'+ "_2"  #41 表示4 0.4 1 0.1
    algo = 'ppo'
    # 存放在sb3model/文件夹下
    save_path = f"trained_model/{env_id}/{model_name}{algo}_{env_id}"

    # tensorboard log 路径
    tensorboard_log_path = f"tensorboard_log/{env_id}/"
    tensorboard_log_name = f"{model_name}{algo}_{env_id}"


                   
    # Instantiate and wrap the environment
    env = make_vec_env(env_id = env_id, n_envs = 15)


    # Create the evaluation environment and callbacks
    eval_env = Monitor(gym.make(env_id))

    callbacks = [EvalCallback(eval_env, best_model_save_path=save_path)]

    RLalgo = {
        "sac": SAC,
        "td3": TD3,
        "ppo": PPO,
    }[algo]

    hyperparams = {
        "sac": dict(
            batch_size=256,
            gamma=0.98,
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_starts=10000,
            buffer_size=int(3e5),
            tau=0.01,
        ),
        "ppo": dict(
            batch_size=512,
            learning_rate=2.5e-4,
            gamma=0.98
        )
    }[algo]

    begin_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    model = RLalgo("MlpPolicy", env, verbose=1, tensorboard_log = tensorboard_log_path, **hyperparams,seed = seed)
    try:
        model.learn(n_timesteps, callback=callbacks , tb_log_name = tensorboard_log_name )
    except KeyboardInterrupt:
        pass
    print('=====================================')
    print(f"Saving to {save_path}.zip")
    model.save(save_path)
    end_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    print('Started at: ' + begin_time)
    print('Ended at: ' + end_time)
    print('=====================================')


if __name__ == '__main__':
    main()