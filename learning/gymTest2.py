'''
gym 调用mujoco包的env
python learning/gymTest2.py
'''
# ------- 来自于mujoco200在win+py3.9下的矫情的要求 --------
# 手动添加mujoco路径
import os
from getpass import getuser
user_id = getuser()
os.add_dll_directory(f"C://Users//{user_id}//.mujoco//mujoco200//bin")
os.add_dll_directory(f"C://Users//{user_id}//.mujoco//mujoco-py-2.0.2.0//mujoco_py")
# -------------------------------------------------------
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import gym
import pybullet_envs
import gym_custom_env  #注册自定义的环境
from time import sleep

print(gym.__version__)
#env = gym.make('HalfCheetah-v2')
#env = gym.make('Humanoid-v3')            # mujoco的env
#env = gym.make('HumanoidBulletEnv-v0')   # bullet的env
env = gym.make('HumanoidCustomEnv-v0', healthy_reward = 3.0)    # 自定义的mujoco env


env.reset()
print(os.path.dirname(__file__))
for _ in range(10000):
    sleep(0.05)
    env.render()                            # 渲染
    act = env.action_space.sample()         # 在动作空间中随机采样
    obs, reward, done, _ = env.step(act)    # 与环境交互
    if done:
        env.reset()
env.close()