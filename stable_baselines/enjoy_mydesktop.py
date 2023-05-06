# 由于我的笔记本显存不够，创建model时会报错，故使用set parameters
# 手动更改模型路径
'''
python stable_baselines/enjoy_mydesktop.py --algo sac --env HumanoidCustomEnv-v0  --terrain-type steps  --model-name 5e6_steps_ankle_t4_cpu8
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
    ################################################################################
    ################################################################################
    # save_path = 'best_model/5e6_steps_t5_cpu8_sac_HumanoidCustomEnv-v0.zip'
    save_path = 'best_model/5e6_evo_steps_t3_cpu8_sac_HumanoidCustomEnv-v0.zip'
    # 更新参数
    params = {   'thigh_lenth':0.3512,           # 大腿长 0.34
            'shin_lenth':0.2572,              # 小腿长 0.2572
            'upper_arm_lenth':0.2343,        # 大臂长 0.2771
            'lower_arm_lenth':0.3282,        # 小臂长 0.2944
            'foot_lenth':0.2105,       }     # 脚长   0.18
    ################################################################################
    ################################################################################
    env_id = args.env
    terrain = args.terrain_type
    # Create an env similar to the training env
    env = gym.make(env_id, terrain_type=terrain)

    env.update_xml_model(params)

    # Enable GUI
    if not args.no_render:
        env.render(mode="human")

    algo = {
        "sac": SAC,
        "td3": TD3,
        "ppo": PPO,
    }[args.algo]

        # Tuned hyperparameters from https://github.com/DLR-RM/rl-baselines3-zoo
    hyperparams = {
        "sac": dict(
            batch_size=256,
            gamma=0.98,
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_starts=10000,
            buffer_size=int(5e4),
            tau=0.01,
            gradient_steps=4,
        ),
        "ppo": dict(
            batch_size=512,
            learning_rate=2.5e-4,
            policy_kwargs=dict(net_arch=({'pi':[128,128]},{'vf':[128,128]})),
            gamma=0.99
        )
    }[args.algo]
    # We assume that the saved model is in the same folder




    print('load from:')
    print(save_path)

    model = algo("MlpPolicy", env, verbose=1,  **hyperparams)
    # Load the saved model
    model.set_parameters(save_path)
    print("==============================")
    print(f"Method: {args.algo}")
    print(f"Time steps: {args.model_name}")
    # print(f"gradient steps:{model.gradient_steps}")
    print("model path:"+save_path)
    print("==============================")

    try:
        # Use deterministic actions for evaluation
        episode_rewards, episode_lengths = [], []
        for i in range(7):
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
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            print(
                f"Episode {len(episode_rewards)} reward={episode_reward}, length={episode_length}"
            )
            print("contact pairs",info["contact pairs"])
            print('forward R: ', forward_r_total)
            print('contact R: ', contact_r_total)
            print('posture R: ', posture_r_total)
            print('healthy R: ', healthy_r_total)
            print('control C: ', control_c_total)
            print('contact C: ', contact_c_total)
            print('is healthy' ,info["is_healthy"])
            print('is walking' ,info["is_walking"])
            print('************************')
            # env.update_xml_model(params)
            # params['shin_lenth'] = 0.5 + 0.1*i

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        mean_len, std_len = np.mean(episode_lengths), np.std(episode_lengths)
        print("========== Results ===========")
        print(f"Episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"Episode_length={mean_len:.2f} +/- {std_len:.2f}")
        print("==============================")
    except KeyboardInterrupt:
        pass

    # Close process
    env.close()
