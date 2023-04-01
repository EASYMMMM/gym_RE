# Code adapted from https://github.com/DLR-RM/rl-baselines3-zoo
# it requires stable-baselines3 to be installed
# Colab Notebook: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/pybullet.ipynb
# You can run it using: python -m pybullet_envs.stable_baselines.enjoy --algo td3 --env HalfCheetahBulletEnv-v0
# Author: Antonin RAFFIN
# MIT License

# 通过info检查，调试训练好的模型。
'''
python stable_baselines/checkModel.py --algo td3 --env HalfCheetah-v2

python stable_baselines/checkModel.py --algo sac --env Humanoid-v3

python stable_baselines/checkModel.py --algo ppo --env Humanoid-v3  --model-name 2e6 

python stable_baselines/checkModel.py --algo sac --env HumanoidCustomEnv-v0   --model-name 2e6 
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
        help="Name of the model",
        default="",
        type=str,
    )
    parser.add_argument(
        "--model-path",
        help="Save path of the model",
        default="",
        type=str,
    )
    args = parser.parse_args()

    env_id = args.env
    # Create an env similar to the training env
    env = gym.make(env_id)

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
    save_path = f"sb3model/{env_id}/{model_name}{args.algo}_{env_id}/best_model.zip"

    if not os.path.isfile(save_path) or args.load_best:
        print("Loading best model")
        # Try to load best model
        save_path = os.path.join(f"{args.algo}_{env_id}", "best_model.zip")

    # Load the saved model
    model = algo.load(save_path, env=env)


    print("==============================")
    print(f"Method: {args.algo}")
    print(f"Time steps: {args.model_name}")
    print("model path:"+save_path)
    print("==============================")
    try:
        # Use deterministic actions for evaluation
        episode_rewards, episode_lengths, end_info = [], [], [] 
        for i in range(args.n_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0
            coms = deque(maxlen=600)
            j = 0
            while not done:  # step 循环开始
                j = j+1
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward

                episode_length += 1
                if not args.no_render:
                    env.render(mode="human")
                    dt = 1.0 / 240.0
                    time.sleep(dt)
                
                print(env.sim.data.qpos[2])
                # 参考用标定点
                env.viewer.add_marker(pos=[0,0,1.0], size=np.array([0.05, 0.05, 0.05]), rgba=np.array([0, 0, 1.0, 1]), type=const.GEOM_SPHERE)
                env.viewer.add_marker(pos=[0,0,2.0], size=np.array([0.05, 0.05, 0.05]), rgba=np.array([0, 0, 1.0, 1]), type=const.GEOM_SPHERE)
                # 躯干跟踪点
                if j % 4 == 0:
                    #coms.append(info['xyz_position'])
                    coms.append(np.array([info["x_position"], info["y_position"] , info["z_position"]  ]))
                    #coms.append(np.array(env.sim.data.qpos[0:3]))
                for com in coms:
                    env.viewer.add_marker(pos=com, size=np.array([0.01, 0.01, 0.01]), rgba=np.array([1., 0, 0, 1]), type=const.GEOM_SPHERE)
                
            end_info.append(info)
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            print(
                f"Episode {len(episode_rewards)} reward={episode_reward}, length={episode_length}"
            )
            del coms


        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        mean_len, std_len = np.mean(episode_lengths), np.std(episode_lengths)
        print("========== Results ===========")
        for i in range(len(episode_rewards)):
            print(f"Episode {i+1}: reward={episode_rewards[i]}, length={episode_lengths[i]}, is walking: {end_info[i]['is_walking']}, is healthy: {end_info[i]['is_healthy']}")
        print(f"Episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"Episode_length={mean_len:.2f} +/- {std_len:.2f}")
        print("==============================")
    except KeyboardInterrupt:
        pass

    # Close process
    env.close()
