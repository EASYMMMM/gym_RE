# Code adapted from https://github.com/DLR-RM/rl-baselines3-zoo
# it requires stable-baselines3 to be installed
# Colab Notebook: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/pybullet.ipynb
# You can run it using: python -m pybullet_envs.stable_baselines.enjoy --algo td3 --env HalfCheetahBulletEnv-v0
# Author: Antonin RAFFIN
# MIT License

'''

python stable_baselines/enjoy_ladder.py --algo sac --env HumanoidCustomEnv-v0  --terrain-type steps 

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
        "--env", type=str, default="HumanoidLadderCustomEnv-v0", help="environment ID"
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


    args = parser.parse_args()

    env_id = args.env


    # Create an env similar to the training env
    # env = gym.make(env_id, terrain_type=terrain)
    env = gym.make(env_id, terrain_type='ladders') 

    # 进化
    #evo_s1
    params = {   'thigh_lenth':0.3321,           # 大腿长 0.34
                'shin_lenth':0.2968,              # 小腿长 0.3
                'upper_arm_lenth':0.2922,        # 大臂长 0.2771
                'lower_arm_lenth':0.2747,        # 小臂长 0.2944
                'foot_lenth':0.2013,       }     # 脚长   0.18
    
    # 楼梯高度升高刺激训练
    params = {'steps_height': 0.30}

    # 更新xml模型
    #env.update_xml_model(params)

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
    # 变重力
    # save_path = 'sb3model//HumanoidLadderCustomEnv-v0//NoGrav_ladder_t1_cpu10_sac_Ladder.zip'
    # save_path = 'sb3model//HumanoidLadderCustomEnv-v0//s1_cpu10_gravity_sac_Ladder.zip'
    # save_path = 'sb3model//HumanoidLadderCustomEnv-v0//s1_cpu10_gravity_ppo_Ladder.zip'
    # No Legs
    # save_path = 'sb3model//HumanoidLadderCustomEnv-v0//ladder_t1_noleg_cpu10_sac_Ladder//best_model.zip'
    # save_path = 'sb3model//HumanoidLadderCustomEnv-v0//ladder_t2_noleg_cpu10_sac_Ladder.zip'
    save_path = 'sb3model//HumanoidLadderCustomEnv-v0//ladder_t3_noleg_cpu10_sac_Ladder.zip'

    print(save_path)

    # Load the saved model
    model = algo.load(save_path, env=env)
    hyperparams =dict(
        batch_size=256,
        gamma=0.98,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_starts=10000,
        buffer_size=int(10000),
        tau=0.01,
        gradient_steps=4,
    )
    #model = SAC("MlpPolicy", env, verbose=1, **hyperparams,seed=1)
    #model.set_parameters(save_path)


    print("==============================")
    print(f"Method: {args.algo}")
    #print(f"Time steps: {args.model_name}")
    # print(f"gradient steps:{model.gradient_steps}")
    print("model path:"+save_path)
    print("==============================")
    try:
        # Use deterministic actions for evaluation
        episode_rewards, episode_lengths, episode_ave_velocitys, episode_success_rate = [], [], [], []
        for _ in range(5):
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
