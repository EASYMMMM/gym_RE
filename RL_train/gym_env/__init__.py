import gym
from gym.envs.registration import registry, make, spec

def register(id, *args, **kvargs):
  if id in registry.env_specs:
    return
  else:
    return gym.envs.registration.register(id, *args, **kvargs)

# ----------- 自定义的env环境 -------------

register(  
    id='InvertedPendulumEnv-v0',
    entry_point='gym_env.InvertedPendulum:InvertedPendulum',
    max_episode_steps=2000,  # dt=0.005s
    reward_threshold=5000.0,
)
