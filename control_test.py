# Code adapted from https://github.com/DLR-RM/rl-baselines3-zoo
# it requires stable-baselines3 to be installed
# Colab Notebook: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/pybullet.ipynb
# You can run it using: python -m pybullet_envs.stable_baselines.enjoy --algo td3 --env HalfCheetahBulletEnv-v0
# Author: Antonin RAFFIN
# MIT License

'''
python control_test.py --algo ppo --env TranslationOscillatorEnv-v0  --model-name t1

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
from gym_custom_env.TranslationOscillator import play
from stable_baselines3 import SAC, TD3, PPO


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Enjoy an RL agent trained using Stable Baselines3"
    )
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
        "--load-best",
        action="store_true",
        default=False,
        help="Load best model instead of last model if available",
    )
    parser.add_argument(
        "--model-name",
        help="Name of the model's save path",
        default="",
        type=str,
    )    
    args = parser.parse_args()

    env_id = 'TranslationOscillatorEnv-v0'

    # Create an env similar to the training env
    #env = gym.make(env_id) 
    env = gym.make(env_id,
                   suqare_reward=True ,
                   #acc_state=True, 
                   reward_weight = [4,0.4,1,0.1])
    
    algo = {
        "sac": SAC,
        "td3": TD3,
        "ppo": PPO,
    }[args.algo]

    # We assume that the saved model is in the same folder


     # 存放在sb3model/文件夹下
    #save_path = f"sb3model/{env_id}/{model_name}{args.algo}_{env_id}.zip"
    #save_path = 'sb3model\\TranslationOscillatorEnv-v0\\t1_Square_ppo_TranslationOscillatorEnv-v0.zip'
    model_name =  't1_wr41_Square_acc_sr1_0init_ppo_'
    save_path = 'sb3model\\TranslationOscillatorEnv-v0\\'+ model_name + 'TranslationOscillatorEnv-v0.zip'
    
    print('load from:')

    print(save_path)
    # Load the saved model
    model = algo.load(save_path, env=env)


    print("==============================")
    print(f"Method: {args.algo}")
    print(f"Time steps: {args.model_name}")
    # print(f"gradient steps:{model.gradient_steps}")
    print("model path:"+save_path)
    print("==============================")
    try:
        # Use deterministic actions for evaluation
        #play(env,model,init_state=[1,1,1,1])
        play(env,model,csv_path='TORA_DATA/'+model_name+'.csv')


    except KeyboardInterrupt:
        pass

    # Close process
    env.close()
