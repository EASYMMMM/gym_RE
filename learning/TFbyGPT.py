'''
我问chatGPT，能不能给一个基于gym和tensorflow的强化学习例程？

chatGPT说：
'''
# python learning/TFbyGPT.py


import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print('### Create by chatGPT ###')
# 创建环境
env = gym.make('CartPole-v0')

# 确定环境状态和行动的形状
state_shape = env.observation_space.shape
action_shape = env.action_space.n

# 创建模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=state_shape),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(action_shape, activation='linear')
])
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

# 设置超参数
num_episodes = 1000
max_steps_per_episode = 1000
discount_rate = 0.95

# 存储回报
rewards = []

# 训练模型
for episode in range(num_episodes):
    state = env.reset()
    episode_rewards = 0
    for step in range(max_steps_per_episode):
        # 执行随机行动
        action = np.random.choice(action_shape)
        next_state, reward, done, info = env.step(action)
        episode_rewards += reward
        if done:
            break
        state = next_state
    rewards.append(episode_rewards)
    print('reward:',episode_rewards) #这句话不是ChatGPT写的

# 打印平均回报
print("Average reward over all episodes: {:.2f}".format(np.mean(rewards)))

# 不是chatGPT写的，打印reward变化
plt.plot(range(len(rewards)),rewards)
plt.show()
# 如果vscode不显示图像，在设置里搜索jupyter.themeMatplotlibPlots，打勾。