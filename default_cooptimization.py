# Code adapted from https://github.com/DLR-RM/rl-baselines3-zoo
# it requires stable-baselines3 to be installed
# Colab Notebook: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/pybullet.ipynb
# You can run it using: python -m pybullet_envs.stable_baselines.train --algo td3 --env HalfCheetahBulletEnv-v0
# Author: Antonin RAFFIN
# MIT License

'''
平坦地形 联合优化

python cooptimization.py --algo td3 --env HalfCheetah-v2

python stable_baselines/multiEnvTrain.py --algo sac --env Humanoid-v3  --model-name 2e6

python stable_baselines/multiEnvTrain.py --algo ppo --env Humanoid-v3  --n-timesteps 2000000 --model-name 2e6

python stable_baselines/multiEnvTrain.py --algo ppo --env HumanoidCustomEnv-v0 --num-cpu 2 --n-timesteps 2000000 --model-name 2e6_t4 

python cooptimization.py --algo sac --env HumanoidCustomEnv-v0 --num-cpu 8 --n-timesteps 5000000 --model-name 5e6_evo_steps_t0 --terrain-type steps 

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
from stable_baselines.rewardChecker import update_info_buffer,dump_logs
from stable_baselines.evolutionCallback import EvolutionCallback

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


    env_id = 'HumanoidCustomEnv-v0'
    num_cpu = 10
    n_timesteps = 1500000
    terrain_type = 'default'
    pretrained_model = 'sb3model\\default_evo_exp\\flatfloor_pretrain_1e6_s2.zip'
    buffer_model = 'sb3model\\default_evo_exp\\flatfloor_pretrain_1e6_s2replay_buffer.pkl'
    tensorboard_log_path = 'experiments\\flat_floor_evo_s2'
    # 添加日志中的reward分析功能
    BaseAlgorithm._update_info_buffer = update_info_buffer
    OffPolicyAlgorithm._dump_logs = dump_logs
    # env kwargs
    env_kwargs = {'terrain_type':terrain_type}

    pretrain = False

    ####################################################################################
    ## PRE TRAIN
    
    turn = 's2t1'
    env_id = 'HumanoidCustomEnv-v0'
    num_cpu = 10
    n_timesteps = 1000000
    model_name = "flatfloor_pretrain_1e6_"+turn

    # 存放在sb3model/文件夹下
    save_path = f"sb3model/default_evo_exp/"+model_name

    # tensorboard log 路径
    tensorboard_log_name = model_name

    # Instantiate and wrap the environment
    env = make_vec_env(env_id = env_id, n_envs = num_cpu, env_kwargs = env_kwargs)

    # Create the evaluation environment and callbacks
    eval_env = Monitor(gym.make(env_id,terrain_type = terrain_type))

    callbacks = [EvalCallback(eval_env, best_model_save_path=save_path)]
    '''callbacks  = [EvolutionCallback(eval_env,n_timesteps,
                                    warm_up_steps=400000,
                                    design_update_steps=100000,
                                    pop_size = 40,
                                    terrain_type = 'default',
                                    pretrain_num=1000000)]'''

    n_actions = env.action_space.shape[0]


    hyperparams =dict(
            batch_size=256,
            gamma=0.98,
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_starts=10000,
            buffer_size=int(50000),
            tau=0.01,
            gradient_steps=4,
        )


    begin_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    

    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log = tensorboard_log_path, **hyperparams)

    if pretrain:
        try:
            model.learn(n_timesteps, callback=callbacks , tb_log_name = tensorboard_log_name )
        except KeyboardInterrupt:
            pass
        print('=====================================')
        print(f"Saving to {save_path}.zip")
        model.save(save_path)
        model.save_replay_buffer(save_path+'replay_buffer')
        end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print('Started at: ' + begin_time)
        print('Ended at: ' + end_time)
        print('=====================================')
    del env
    del model
    del callbacks

    pretrained_model = 'sb3model\\default_evo_exp\\flatfloor_pretrain_1e6_s2.zip'
    buffer_model = 'sb3model\\default_evo_exp\\flatfloor_pretrain_1e6_s2replay_buffer.pkl'
    ####################################################################################
    ## 无惩罚 EVO

    model_name = "flatfloor_evo_"+turn
    # 模型存放路径
    save_path = f"sb3model/default_evo_exp/{model_name}"
    # tensorboard log 文件名称
    tensorboard_log_name = model_name
    # Instantiate and wrap the environment
    env = make_vec_env(env_id = env_id, n_envs = num_cpu, env_kwargs = env_kwargs)

    # Create the evaluation environment and callbacks
    eval_env = Monitor(gym.make(env_id,terrain_type = terrain_type))
    # callbacks = [EvalCallback(eval_env, best_model_save_path=save_path)]
    callbacks  = [EvolutionCallback(eval_env,n_timesteps,
                                    warm_up_steps=400000,
                                    design_update_steps=100000,
                                    pop_size = 40,
                                    terrain_type = 'default',
                                    pretrain_num=1000000)]

    begin_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    
    # 加载预训练模型
    model = SAC.load(pretrained_model,
                    env = env,
                    tensorboard_log = tensorboard_log_path
                     )
    # 加载buffer
    model.load_replay_buffer(buffer_model)

    try:
        model.learn(n_timesteps, callback=callbacks ,reset_num_timesteps = False, tb_log_name = tensorboard_log_name )
    except KeyboardInterrupt:
        pass
    print('=====================================')
    print(f"Saving to {save_path}.zip")
    model.save(save_path)
    end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print('Started at: ' + begin_time)
    print('Ended at: ' + end_time)
    print('=====================================')

    del env
    del model
    del callbacks

    ####################################################################################
    ## 有惩罚1 EVO

    model_name = "flatfloor_evo_30punish_"+turn
    # 模型存放路径
    save_path = f"sb3model/default_evo_exp/{model_name}"
    # tensorboard log 文件名称
    tensorboard_log_name = model_name
    # Instantiate and wrap the environment
    env = make_vec_env(env_id = env_id, n_envs = num_cpu, env_kwargs = env_kwargs)

    # Create the evaluation environment and callbacks
    eval_env = Monitor(gym.make(env_id,terrain_type = terrain_type))
    # callbacks = [EvalCallback(eval_env, best_model_save_path=save_path)]
    callbacks  = [EvolutionCallback(eval_env,n_timesteps,
                                    warm_up_steps=400000,
                                    design_update_steps=100000,
                                    pop_size = 40,
                                    terrain_type = 'default',
                                    pretrain_num=1000000,
                                    overchange_punish=1)]

    begin_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    
    # 加载预训练模型
    model = SAC.load(pretrained_model,
                    env = env,
                    tensorboard_log = tensorboard_log_path
                     )
    # 加载buffer
    model.load_replay_buffer(buffer_model)

    try:
        model.learn(n_timesteps, callback=callbacks ,reset_num_timesteps = False, tb_log_name = tensorboard_log_name )
    except KeyboardInterrupt:
        pass
    print('=====================================')
    print(f"Saving to {save_path}.zip")
    model.save(save_path)
    end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print('Started at: ' + begin_time)
    print('Ended at: ' + end_time)
    print('=====================================')

    del env
    del model
    del callbacks

    ####################################################################################
    ## 原模型继续训练

    model_name = "flatfloor_noevo_"+turn
    # 模型存放路径
    save_path = f"sb3model/default_evo_exp/{model_name}"
    # tensorboard log 文件名称
    tensorboard_log_name = model_name
    # Instantiate and wrap the environment
    env = make_vec_env(env_id = env_id, n_envs = num_cpu, env_kwargs = env_kwargs)

    # Create the evaluation environment and callbacks
    eval_env = Monitor(gym.make(env_id,terrain_type = terrain_type))
    callbacks = [EvalCallback(eval_env, best_model_save_path=save_path)]
    '''callbacks  = [EvolutionCallback(eval_env,n_timesteps,
                                    warm_up_steps=400000,
                                    design_update_steps=100000,
                                    pop_size = 40,
                                    terrain_type = 'default',
                                    pretrain_num=1000000,
                                    overchange_punish=30)]'''

    begin_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    
    # 加载预训练模型
    model = SAC.load(pretrained_model,
                    env = env,
                    tensorboard_log = tensorboard_log_path
                     )
    # 加载buffer
    model.load_replay_buffer(buffer_model)

    try:
        model.learn(n_timesteps, callback=callbacks ,reset_num_timesteps = False, tb_log_name = tensorboard_log_name )
    except KeyboardInterrupt:
        pass
    print('=====================================')
    print(f"Saving to {save_path}.zip")
    model.save(save_path)
    end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print('Started at: ' + begin_time)
    print('Ended at: ' + end_time)
    print('=====================================')

    del env
    del model
    del callbacks