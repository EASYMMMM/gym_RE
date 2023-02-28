'''
gym 测试
生成一个倒立摆

python learning/gymTest0.py
'''


import gym
print(gym.__version__)
env = gym.make('CartPole-v1')
env.reset()

for _ in range(10000):
    env.render()                            # 渲染
    act = env.action_space.sample()         # 在动作空间中随机采样
    obs, reward, done, _ = env.step(act)    # 与环境交互
    if done:
        env.reset()
env.close()