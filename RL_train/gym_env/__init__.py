import gym
from gym.envs.registration import registry, make, spec
from gym_custom_env.generateXML import HumanoidXML
def register(id, *args, **kvargs):
  if id in registry.env_specs:
    return
  else:
    return gym.envs.registration.register(id, *args, **kvargs)

# ----------- 自定义的env环境 -------------

register(  
    id='InvertedPendulumEnv-v0',
    entry_point='gym_custom_env.InvertedPendulum:InvertedPendulum',
    max_episode_steps=10000,
    reward_threshold=500.0,
)
