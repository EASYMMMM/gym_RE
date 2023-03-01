'''
gym 调用mujoco包的env
python learning/gymTest2.py
'''
import os

#os.add_dll_directory("C://Users//zdh//.mujoco//mujoco200//bin")
#os.add_dll_directory("C://Users//zdh//.mujoco//mujoco-py-2.0.2.0//mujoco_py")
os.add_dll_directory("C://Users//孟一凌//.mujoco//mujoco200//bin")
os.add_dll_directory("C://Users//孟一凌//.mujoco//mujoco-py-2.0.2.0//mujoco_py")
import gym
import pybullet_envs
print(gym.__version__)
#env = gym.make('HalfCheetah-v2')
env = gym.make('Humanoid-v3')

#env = gym.make('HumanoidBulletEnv-v0')
env.reset()

for _ in range(10000):
    env.render()                            # 渲染
    act = env.action_space.sample()         # 在动作空间中随机采样
    obs, reward, done, _ = env.step(act)    # 与环境交互
    if done:
        env.reset()
env.close()