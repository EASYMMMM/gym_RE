'''
python pendulum_test.py 
'''
# ------- 来自于mujoco150在win+py3.9下的矫情的要求 --------
# 手动添加mujoco路径
import os
from getpass import getuser
import time
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import gym_env       # 注册自定义环境
import gym
import numpy as np
from gym_env.InvertedPendulum import pendulum_animation, save_gif
from stable_baselines3 import SAC, TD3, PPO
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Enjoy an RL agent trained using Stable Baselines3"
    ) 
    parser.add_argument(
        "--model",
        help="model name",
        type=str,
    )
    args = parser.parse_args()

    # 随机种子
    seed = 1

    # 环境名
    env_id = 'InvertedPendulumEnv-v0'

    env_kwargs = { "energy_obs":True,
                   "init_pos":4.18}
    
    # Create an env similar to the training env
    #env = gym.make(env_id) 
    env = gym.make(env_id,**env_kwargs)
    
    algo = {
        "sac": SAC,
        "td3": TD3,
        "ppo": PPO,
    }["ppo"]

    model_name =  'InvPend_2'

    #save_path = 'trained_model/InvertedPendulumEnv-v0/'+ model_name + 'ppo_'+'InvertedPendulumEnv-v0.zip'
    save_path = 'runs/'+args.model+'/InvPend_0ppo_InvertedPendulumEnv-v0.zip'
    
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
        i = 0
        frame_skip=10
        pend_a = []
        Action = []
        pend_adot = []
        pend_energy = []
        while i<3000:
            i = i+1
            action, _ = model.predict(obs, deterministic=True)
            if action== 2:
                print(action)
            obs, reward, done, info = env.step(action)   
            if i%frame_skip == 0:   # 降低绘图帧数
                pend_a.append(obs[0]) 
                Action.append(action)
                pend_adot.append(obs[1])
                pend_energy.append(env.system_energy)
            if done:
                break
        fig, axs = plt.subplots(2,2, )
        
        axs[0][0].plot(pend_a, label = 'theta')
        axs[0][0].legend()
        axs[0][0].grid(True,linestyle = '--')
        axs[0][1].plot(pend_adot, label = 'theta_dot')
        axs[0][1].legend()
        axs[0][1].grid(True,linestyle = '--')
        axs[1][0].plot(Action, label = 'action')
        axs[1][0].legend()
        axs[1][0].grid(True,linestyle = '--')
        axs[1][1].plot(pend_energy, label = 'energy')
        axs[1][1].legend()
        axs[1][1].grid(True,linestyle = '--')

        plt.show()
        plt.savefig('runs/'+args.model+'/result_curve.png', dpi=300)

        animation = pendulum_animation(pend_a)
        save_gif(animation, 'runs/'+args.model+'/pendulum_animation'+model_name+'.gif',fps=200/frame_skip)


    except KeyboardInterrupt:
        pass

    # Close process
    env.close()
