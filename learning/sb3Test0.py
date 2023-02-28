'''
来自 stable baseline3 官方文档
gym
使用stable baseline3训练gym中的倒立摆环境，并展示
python learning/sb3Test0.py
'''

import gym

from stable_baselines3 import A2C

env = gym.make("CartPole-v1")

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()