'''
gymTest 尝试通过pybullet引入gym环境

python learning/gymTest1.py

'''

import pybullet as p
import pybullet_envs
from time import sleep
import gym

cid = p.connect(p.DIRECT)
env = gym.make("HumanoidBulletEnv-v0")
env.render()
env.reset()

for _ in range(10000):
    sleep(1 / 60)
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)

p.disconnect(cid)