'''
https://cloud.tencent.com/developer/article/1619441
强化学习系列案例 | 强化学习实验环境Gym和TensorFlow
python learning/gym_tfTest.py
'''
import gym
from gym import envs
env_spaces = envs.registry.all()
env_ids = [env_space.id for env_space in env_spaces]
print(env_ids)

env = gym.make('CliffWalking-v0') #悬崖寻路环境
print('状态空间:',env.observation_space)
print('动作空间:',env.action_space)

print(env.P[35])

# 初始化环境
env.reset()
'''
step方法，其接受智能体的动作作为参数，并返回以下四个值：
    observation：采取当前动作后到达的状态
    reward：采取当前行动所获得的奖励
    done：布尔型变量，表示是否到达终点状态
    info：字典类型的值，包含一些调试信息，例如状态间的转移概率
'''
for t in range(10):
    # 在动作空间中随机选择一个动作
    action = env.action_space.sample()
    # 采取一个动作
    observation, reward, done, info = env.step(action)
    print("action:{},observation:{},reward:{}, done:{}, info:{}".format(action, observation, reward, done, info))
    # 图形化显示
    env.render()     