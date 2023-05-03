from stable_baselines3 import PPO

tensorboard_log = '/tmp/debug/'
model = PPO('MlpPolicy', 'CartPole-v1', tensorboard_log=tensorboard_log)
# Get the env object
env = model.get_env()
# The mean reward starts from zero
model.learn(50000)
# After 50k steps, it reaches a mean reward > 200
model.save('/tmp/test_ppo')
# Delete the trained model
del model

model = PPO.load('/tmp/test_ppo.pkl', env=env, tensorboard_log=tensorboard_log)
# the mean reward starts from about 200
model.learn(50000)