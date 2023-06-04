# Code adapted from https://github.com/DLR-RM/rl-baselines3-zoo
# it requires stable-baselines3 to be installed
# Colab Notebook: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/pybullet.ipynb
# You can run it using: python -m pybullet_envs.stable_baselines.enjoy --algo td3 --env HalfCheetahBulletEnv-v0
# Author: Antonin RAFFIN
# MIT License

'''
python  stable_baselines/enjoy_mydesktop.py --algo sac --env HumanoidCustomEnv-v0  --terrain-type default

'''
# ------- 来自于mujoco150在win+py3.9下的矫情的要求 --------
# 手动添加mujoco路径
import os
from getpass import getuser
user_id = getuser()
os.add_dll_directory(f"C://Users//{user_id}//.mujoco//mujoco200//bin")
os.add_dll_directory(f"C://Users//{user_id}//.mujoco//mujoco-py-2.0.2.0//mujoco_py")
# -------------------------------------------------------
import time
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import gym_custom_env       # 注册自定义环境
import gym
import numpy as np
import pybullet_envs
from gym.wrappers import Monitor
from stable_baselines3 import SAC, TD3, PPO
from mujoco_py.generated import const
from collections import deque

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
        "-n", "--n-episodes", help="Number of episodes", default=5, type=int
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        default=False,
        help="Do not render the environment",
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
    parser.add_argument(
        "--terrain-type",
        help="Type of the traning terrain",
        default='default',
        type=str,
    )        
    args = parser.parse_args()

    env_id = args.env
    terrain = args.terrain_type
    # Create an env similar to the training env
    env = gym.make(env_id, terrain_type=terrain, flatfloor_size=16, y_limit = False)
    #evo_punish_s3
    params = {   'thigh_lenth':0.3185,           # 大腿长 0.34
                'shin_lenth':0.231,              # 小腿长 0.3
                'upper_arm_lenth':0.3095,        # 大臂长 0.2771
                'lower_arm_lenth':0.2214,        # 小臂长 0.2944
                'foot_lenth':0.1526,       }     # 脚长   0.18
    env.update_xml_model(params)
    # Enable GUI
    if not args.no_render:
        env.render(mode="human")

    algo = {
        "sac": SAC,
        "td3": TD3,
        "ppo": PPO,
    }[args.algo]

    # We assume that the saved model is in the same folder

  
    print('load from:')
    #save_path ='sb3model/default_evo_exp/flatfloor_pretrain_1e6_s2.zip'
    #save_path = 'best_model\\flatfloor_exp_s3\\flatfloor_noevo_s3.zip'
    #save_path = 'sb3model\\default_evo_exp\\flatfloor_evo_s3.zip'
    save_path = 'best_model\\flatfloor_exp_s3\\flatfloor_evo_punish_s3.zip'
    
    #save_path = 'best_model\\steps_evo_exp\\steps_noevo_s1.zip'
    #save_path = 'best_model\\steps_evo_exp\\steps_evo_punish_s1.zip'
    #save_path = 'sb3model\\steps_evo_exp\\steps_evo_s1.zip'

    print(save_path)


    # 加载完整模型
    #model = algo.load(save_path, env=env)

    # 只加载网络
    hyperparams =dict(
        batch_size=256,
        gamma=0.98,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_starts=10000,
        buffer_size=int(10000),
        tau=0.01,
        gradient_steps=4,
    )
    model = SAC("MlpPolicy", env, verbose=1, **hyperparams,seed=1)
    model.set_parameters(save_path)


    print("==============================")
    print(f"Method: {args.algo}")
    print(f"Time steps: {args.model_name}")
    # print(f"gradient steps:{model.gradient_steps}")
    print("model path:"+save_path)
    print("==============================")

    outdir = 'screenshot'
    #env = Monitor(env, outdir, video_callable=lambda episode_id: True,  force=True) 

    try:
        # Use deterministic actions for evaluation
        episode_rewards, episode_lengths, episode_ave_velocitys, episode_success_rate = [], [], [], []
        for _ in range(30):
            obs = env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0
            forward_r_total = 0
            contact_r_total = 0
            posture_r_total = 0
            healthy_r_total = 0
            control_c_total = 0
            contact_c_total = 0
            coms = deque(maxlen=1000)
            j = 0
            while not done:
                j = j+1
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                color = np.array([1.0, 0, 0.0, 1])
                if env.sim.data.qpos[0] > 13: # 平地
                    color = np.array([0,1.0, 0.0, 1])
                if env.sim.data.qpos[0] > 6.3: # 楼梯
                    color = np.array([0,1.0, 0.0, 1])                    
                if not args.no_render:
                    env.render(mode="human")
                    # 平地
                    #env.viewer.add_marker(pos=[13,0.4,1], size=np.array([0.05, 0.05, 1.0]), label="",rgba=color, type=const.GEOM_CYLINDER)
                    #env.viewer.add_marker(pos=[13,-0.4,1], size=np.array([0.05, 0.05, 1.0]), label="",rgba=color, type=const.GEOM_CYLINDER)
                    # 楼梯
                    env.viewer.add_marker(pos=[6.3,1.0,3.2], size=np.array([0.05, 0.05, 1.0]), label="",rgba=color, type=const.GEOM_CYLINDER)
                    env.viewer.add_marker(pos=[6.3,-1.0,3.2], size=np.array([0.05, 0.05, 1.0]), label="",rgba=color, type=const.GEOM_CYLINDER)

                    dt = 1.0 / 240.0
                    time.sleep(dt)
                # 躯干跟踪点
                if j % 3 == 0:
                    #coms.append(info['xyz_position'])
                    #coms.append(np.array([info["x_position"], info["y_position"] , info["z_position"]  ]))
                    coms.append(np.array(env.sim.data.qpos[0:3]))
                #for com in coms:
                #    env.viewer.add_marker(pos=com, size=np.array([0.06, 0.06, 0.06]), label="",rgba=np.array([0, 0, 1, 1]), type=const.GEOM_SPHERE)
            time.sleep(1)


            detail = info.get('reward_details')
            forward_r_total += detail['forward_reward_sum']
            final_x          = info['xyz_position'][0]
            ave_velocity     = info['ave_velocity']
            is_success       = info['is_success']
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_ave_velocitys.append(ave_velocity)
            episode_success_rate.append(is_success)
            print(
                f"Episode {len(episode_rewards)} reward={episode_reward}, length={episode_length}"
            )
            print('forward R: ', forward_r_total)
            print('final x:',final_x)
            print('ave velocity:',ave_velocity)
            print('success:',is_success)
            print('************************')

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        mean_len, std_len = np.mean(episode_lengths), np.std(episode_lengths)

        mean_ave_v = np.mean(episode_ave_velocitys)
        std_ave_v = np.std(episode_ave_velocitys)
        success_rate = sum(episode_success_rate)/len(episode_success_rate)
        print("========== Results ===========")
        print(f"Episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"Episode_length={mean_len:.2f} +/- {std_len:.2f}")
        print(f"Episode_ave_v={mean_ave_v:.2f} +/- {std_ave_v:.2f}")
        print(f"Episode_success_rate={success_rate:.2f}")
        print("==============================")
    except KeyboardInterrupt:
        pass

    # Close process
    env.close()
