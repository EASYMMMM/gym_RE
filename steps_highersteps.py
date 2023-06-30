'''
楼梯层高升高，刺激智能体学习

python steps_highersteps.py 

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

def update_xml_model(self,params):
    # VecEnv更新XML模型
    for env_idx in range(self.num_envs):
        self.envs[env_idx].update_xml_model(params)
DummyVecEnv.update_xml_model = update_xml_model

if __name__ == "__main__":

    # 添加日志中的reward分析功能
    BaseAlgorithm._update_info_buffer = update_info_buffer
    OffPolicyAlgorithm._dump_logs = dump_logs

    # 随机种子
    seed = 1

    env_id = 'HumanoidCustomEnv-v0'
    num_cpu = 10
    n_timesteps = 2000000
    model_name = f"steps_gethigher_s{seed}"
    
    terrain_type = 'steps'
    env_kwargs = {'terrain_type':terrain_type}

    # 存放在sb3model/文件夹下
    save_path = "sb3model/steps_gethigher/"+model_name

    # tensorboard log 路径
    tensorboard_log_path = f'experiments_data\\steps_gethigher_s{seed}'
    tensorboard_log_name = model_name

    height_list = list()
    ###############################################################################
    # pretrain

    # Instantiate and wrap the environment
    env = make_vec_env(env_id = env_id, n_envs = num_cpu, env_kwargs = env_kwargs)
    # Create the evaluation environment and callbacks
    eval_env = Monitor(gym.make(env_id,terrain_type = terrain_type))
    callbacks = [EvalCallback(eval_env, best_model_save_path=save_path)]

    n_actions = env.action_space.shape[0]

    # Tuned hyperparameters from https://github.com/DLR-RM/rl-baselines3-zoo
    hyperparams =dict(
            batch_size=256,
            gamma=0.98,
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_starts=10000,
            buffer_size=int(1000000),
            tau=0.01,
            gradient_steps=4,
        )


    begin_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    
    steps_H = 0.10
    height_list.append(steps_H)
    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log = tensorboard_log_path, **hyperparams,seed=seed)

    try:
        model.learn(n_timesteps, callback=callbacks , tb_log_name = tensorboard_log_name )
    except KeyboardInterrupt:
        pass


    ####################################################################################
    ## 更改楼梯高度继续训练
    for i in range(3):
        steps_H = steps_H + 0.05
        height_list.append(steps_H)
        design_params = {'steps_height': steps_H}
        model.env.update_xml_model(design_params)
        tensorboard_log_name = model_name+'_'+str(steps_H) 
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
    print('Steps Height: '+str(height_list))
    print('=====================================')


 