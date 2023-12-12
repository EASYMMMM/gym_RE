'''
tensorflow 安装测试
python learning/tfTest0.py
'''
import torch 
if torch.cuda.is_available():


    print('hello tf')   
 


'''
import pybullet as p
from time import sleep
from pybullet_envs.bullet import CartPoleBulletEnv

cid = p.connect(p.DIRECT)
env = CartPoleBulletEnv(renders=True, discrete_actions=False)

env.render()
env.reset()

for _ in range(10000):
    sleep(1 / 60)
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)

p.disconnect(cid)
'''