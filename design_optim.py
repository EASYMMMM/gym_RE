from GA import GA_Design_Optim
# ------- 来自于mujoco150在win+py3.9下的矫情的要求 --------
# 手动添加mujoco路径
import os
from getpass import getuser
user_id = getuser()
os.add_dll_directory(f"C://Users//{user_id}//.mujoco//mujoco200//bin")
os.add_dll_directory(f"C://Users//{user_id}//.mujoco//mujoco-py-2.0.2.0//mujoco_py")
# -------------------------------------------------------
import time
import argparse
import sys
import gym_custom_env       # 注册自定义环境
import gym
import numpy as np
import pybullet_envs
from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns # 导入模块
import pandas as pd
sns.set() # 设置美化参数，一般默认就好

def update_xml_model(self,params):
    # VecEnv更新XML模型
    for env_idx in range(self.num_envs):
        self.envs[env_idx].update_xml_model(params)
    

DummyVecEnv.update_xml_model = update_xml_model
print('load from:')
save_path = 'best_model/5e6_steps_t5_cpu8_sac_HumanoidCustomEnv-v0.zip'
print(save_path)
env_kwargs = {'terrain_type':'steps'}
env_model = make_vec_env(env_id = 'HumanoidCustomEnv-v0', n_envs = 5, env_kwargs = env_kwargs)
algo = 'sac'
hyperparams =  dict(
        batch_size=256,
        gamma=0.98,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_starts=10000,
        buffer_size=int(5e2),
        tau=0.01,
        gradient_steps=4,
    )
model = SAC("MlpPolicy", env_model, verbose=1,  **hyperparams)
model.set_parameters(save_path)

GA_optimizer = GA_Design_Optim(model,decode_size = 20,POP_size=50,n_generations=10)

GA_optimizer.evolve()
#GA_optimizer.save_fig(fig_name= 'GA_test1')

pop_data = GA_optimizer.pop_data
fitness_data = GA_optimizer.fitness_data




t,s,u,l,f = GA_optimizer.translateDNA(np.array(GA_optimizer.best_individual))

fig = plt.figure(1,figsize=(8, 8))
plt.subplot(2,1,1)
plt.plot(t,color='r')
plt.plot(s,color='g')
plt.plot(u,color='b')
plt.plot(l,color='c')
plt.plot(f,color='m')
plt.legend(['t','s','u','l','f'])
plt.title('design parameters')
plt.subplot(2,1,2)
plt.plot(GA_optimizer.best_reward,color='b')
plt.title('reward')
fig.tight_layout(pad=0.4, w_pad=0, h_pad=0)
fig_name = 'GA_optim_result'
#plt.savefig('screenshot/'+fig_name+'.png', bbox_inches='tight')
plt.show()