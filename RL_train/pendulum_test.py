'''
python RL_train/pendulum_test.py 
'''
# ------- 来自于mujoco150在win+py3.9下的矫情的要求 --------
# 手动添加mujoco路径
import os
from getpass import getuser
import time
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import gym_custom_env       # 注册自定义环境
import gym
import numpy as np
from gym_custom_env.InvertedPendulum import pendulum_animation, save_gif
from stable_baselines3 import SAC, TD3, PPO


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Enjoy an RL agent trained using Stable Baselines3"
    ) 
    args = parser.parse_args()

    # 随机种子
    seed = 1

    # 环境名
    env_id = 'InvertedPendulumEnv-v0'

    # Create an env similar to the training env
    #env = gym.make(env_id) 
    env = gym.make(env_id,)
    
    algo = {
        "sac": SAC,
        "td3": TD3,
        "ppo": PPO,
    }["ppo"]

    model_name =  'InvPend_1'

    save_path = 'sb3model/InvertedPendulumEnv-v0/'+ model_name + 'ppo_'+'InvertedPendulumEnv-v0.zip'
    
    print('load from:')

    print(save_path)
    # Load the saved model
    model = algo.load(save_path, env=env)


    print("==============================")
    print(f"Method: ppo")
    print(f"Time steps: 0.005")
    # print(f"gradient steps:{model.gradient_steps}")
    print("model path:"+save_path)
    print("==============================")
    try:
        obs = env.reset()
        pend_a = []
        i = 0
        frame_skip=10
        while i<200:
            i = i+1
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)   
            if i%frame_skip == 0:   # 降低绘图帧数
                pend_a.append(obs[0]) 
        animation = pendulum_animation(pend_a)
        save_gif(animation, 'RL_train/pendulum_animation'+model_name+'.gif',fps=200/frame_skip)


    except KeyboardInterrupt:
        pass

    # Close process
    env.close()
