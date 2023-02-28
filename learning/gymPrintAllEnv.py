'''
打印gym版本，及所有可用环境。
python learning/gymPrintAllEnv.py
'''

import gym
from gym import envs

print('=='*20)
print('gym: {}'.format(gym.__version__))
print('=='*20)

for env in envs.registry.all():
    print(env.id)