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
    id='TranslationOscillatorEnv-v0',
    entry_point='gym_custom_env.TranslationOscillator:TranslationOscillator',
    max_episode_steps=2000,
    reward_threshold=2000.0,
)
 
register(  
    id='CircleBoxCustomBulletEnv-v0',
    entry_point='gym_custom_env.circleBox:CircleDrive',
    max_episode_steps=2000,
    reward_threshold=2000.0,
)
 # mujoco环境
register(  
    id='HumanoidCustomEnv-v0',
    entry_point='gym_custom_env.humanoidCustom:HumanoidCustomEnv',
    max_episode_steps=1000,
)

 # mujoco环境
register(  
    id='HumanoidLadderCustomEnv-v0',
    entry_point='gym_custom_env.humanoidLadderCustom:HumanoidLadderCustomEnv',
    max_episode_steps=1000,
)
