# Code adapted from https://github.com/DLR-RM/rl-baselines3-zoo
# it requires stable-baselines3 to be installed
# Colab Notebook: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/pybullet.ipynb
# You can run it using: python -m pybullet_envs.stable_baselines.enjoy --algo td3 --env HalfCheetahBulletEnv-v0
# Author: Antonin RAFFIN
# MIT License

'''
python stable_baselines/enjoy.py --algo td3 --env HalfCheetah-v2

python stable_baselines/enjoy.py --algo sac --env Humanoid-v3

python stable_baselines/enjoy.py --algo ppo --env Humanoid-v3  --model-name 2e6 

python stable_baselines/enjoy.py --algo sac --env HumanoidCustomEnv-v0  --terrain-type ladders  --model-name 2e6_ladder_hand_t4_cpu8


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
    env = gym.make(env_id, terrain_type=terrain)
    #evo_s1
    params = {   'thigh_lenth':0.3806,           # 大腿长 0.34
                'shin_lenth':0.2439,              # 小腿长 0.3
                'upper_arm_lenth':0.2837,        # 大臂长 0.2771
                'lower_arm_lenth':0.3638,        # 小臂长 0.2944
                'foot_lenth':0.2062,       }     # 脚长   0.18
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

    model_name = args.model_name + "_"
     # 存放在sb3model/文件夹下
    save_path = f"sb3model/{env_id}/{model_name}{args.algo}_{env_id}.zip"
    
    if not os.path.isfile(save_path) or args.load_best:
        print("Loading best model")
        # Try to load best model
        save_path = os.path.join(f"{args.algo}_{env_id}", "best_model.zip")
    print('load from:')
    #save_path ='sb3model/default_evo_exp/flatfloor_pretrain_1e6_s2.zip'
    #save_path = 'sb3model\\default_evo_exp\\flatfloor_noevo_s2t1.zip'
    #save_path = 'sb3model\\default_evo_exp\\flatfloor_evo_s2t1.zip'
    #save_path = 'sb3model\\default_evo_exp\\flatfloor_evo_30punish_s2t1.zip'
    
    #save_path = 'sb3model\\steps_evo_exp\\steps_noevo_s1.zip'
    #save_path = 'sb3model\\steps_evo_exp\\steps_evo_punish_s1.zip'
    save_path = 'sb3model\\steps_evo_exp\\steps_evo_s1.zip'
    
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
        episode_rewards, episode_lengths, episode_ave_velocitys, episode_success_rate = [], [], [], []
        for _ in range(10):
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
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                if not args.no_render:
                    env.render(mode="human")
                    dt = 1.0 / 240.0
                    time.sleep(dt)
            detail = info.get('reward_details')
            forward_r_total += detail['forward_reward_sum']
            contact_r_total += detail['contact_reward_sum']
            posture_r_total += detail['posture_reward_sum']
            healthy_r_total += detail['healthy_reward_sum']
            control_c_total += detail['control_cost_sum']
            contact_c_total += detail['contact_cost_sum']
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
            #print("contact pairs",info["contact pairs"])
            print('forward R: ', forward_r_total)
            #print('contact R: ', contact_r_total)
            #print('posture R: ', posture_r_total)
            #print('healthy R: ', healthy_r_total)
            #print('control C: ', control_c_total)
            #print('contact C: ', contact_c_total)
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
