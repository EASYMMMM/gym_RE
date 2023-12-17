# Code adapted from https://github.com/DLR-RM/rl-baselines3-zoo
# it requires stable-baselines3 to be installed
# Colab Notebook: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/pybullet.ipynb
# You can run it using: python -m pybullet_envs.stable_baselines.train --algo td3 --env HalfCheetahBulletEnv-v0
# Author: Antonin RAFFIN
# MIT License

'''
使用服务器进行训练 cuda1


python control_train.py --algo ppo --env TranslationOscillatorEnv-v0 --n-timesteps 2000000 --model-name t3

python control_train_weight.py 
'''
import argparse

# ------- 来自于mujoco150在win+py3.9下的矫情的要求 --------
# 手动添加mujoco路径
import os
from getpass import getuser
user_id = getuser()
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
from stable_baselines3.common.env_util import make_vec_env

def main():
    parser = argparse.ArgumentParser("Train an RL agent using Stable Baselines3")

    # 随机种子
    seed = 1

    # 环境名
    env_id = 'TranslationOscillatorEnv-v0'
    n_timesteps = 3000000
    model_name = 't3_wr41_Square_acc_sr05_randinit'+ "_"  #41 表示4 0.4 1 0.1
    algo = 'sac'
    # 存放在sb3model/文件夹下
    save_path = f"sb3model/{env_id}/{model_name}{algo}_{env_id}"

    # tensorboard log 路径
    tensorboard_log_path = f"tensorboard_log/{env_id}/t3/"
    tensorboard_log_name = f"{model_name}{algo}_{env_id}"

    env_kwargs = { "suqare_reward":True ,
                   "acc_state":True, 
                   "init_state" : [0,0,0,0],
                   "stable_reward":  2,
                   "stable_limit" : 0.05,
                   "random_init" : True,
                   "reward_weight": [4,0.4,1,0.1]}
                   
    # Instantiate and wrap the environment
    env = make_vec_env(env_id = env_id, n_envs = 12, env_kwargs = env_kwargs)

    # Create the evaluation environment and callbacks
    eval_env = Monitor(gym.make(env_id, 
                                suqare_reward=True ,
                                acc_state=True, 
                                stable_reward = 2,
                                stable_limit = 0.1,
                                random_init = True,
                                reward_weight = [4,0.4,1,0.1]))

    callbacks = [EvalCallback(eval_env, best_model_save_path=save_path)]

    RLalgo = {
        "sac": SAC,
        "td3": TD3,
        "ppo": PPO,
    }[algo]

    n_actions = env.action_space.shape[0]
    hyperparams = {
        "sac": dict(
            batch_size=256,
            gamma=0.98,
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_starts=10000,
            buffer_size=int(3e5),
            tau=0.01,
            device='cuda:1'
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

    return
#####################################################################################
    # 2倍
    model_name = 't3_wr41_Square_acc_sr05_randinit'+ "_"
    # 存放在sb3model/文件夹下
    save_path = f"sb3model/{env_id}/{model_name}{algo}_{env_id}"

    # tensorboard log 路径
    tensorboard_log_name = f"{model_name}{algo}_{env_id}"

    # Instantiate and wrap the environment
    env = gym.make(env_id, 
                   suqare_reward=True ,
                   acc_state=True, 
                   stable_reward = 2,
                   stable_limit = 0.05,
                   random_init = True,
                   reward_weight = [4,0.4,1,0.1])
    # Create the evaluation environment and callbacks
    eval_env = Monitor(gym.make(env_id, 
                                suqare_reward=True ,
                                acc_state=True, 
                                stable_reward = 2,
                                stable_limit = 0.2,
                                reward_weight = [4,0.4,1,0.1]))
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