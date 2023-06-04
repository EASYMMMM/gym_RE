# 模仿GA
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from getpass import getuser
import sys,os
user_id = getuser()
os.add_dll_directory(f"C://Users//{user_id}//.mujoco//mujoco200//bin")
os.add_dll_directory(f"C://Users//{user_id}//.mujoco//mujoco-py-2.0.2.0//mujoco_py")
# -------------------------------------------------------
import pybullet_envs  # register pybullet envs
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import gym_custom_env       # 注册自定义环境
import time
import gym
import torch
import numpy as np
from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from doctest import master
from sklearn.linear_model import SGDRegressor
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from getpass import getuser
import sys, os
import random

# -------------------------------------------------------




def update_xml_model(self, params):
    # VecEnv更新XML模型
    for env_idx in range(self.num_envs):
        self.envs[env_idx].update_xml_model(params)


DummyVecEnv.update_xml_model = update_xml_model


class GMM_Design_Optim():

    def __init__(
            self,
            model,  # 强化学习模型
            decode_size=24,
            POP_size=80,
            crossover_rate=0.6,
            mutation_rate=0.01,
            n_generations=10,
            n_envs=8,
            optim_bound=[0.7, 1.3],
            overchange_punish=0,  # 更新幅度过大的惩罚项
            elite_num=3,  # 精英策略
            terrain_type='steps',
            num_epochs=100
            ):
        # self.crossover_rate = crossover_rate
        # self.mutation_rate  = mutation_rate
        # self.n_generations  = n_generations                    # 迭代次数
        self.optim_bound = optim_bound
        self.overchange_punish = overchange_punish
        self.elite_num = elite_num
        # self.__origin_design_param1 = {'thigh_length_1': 0.34, 'thigh_length_2': 0.34}
        # self.__origin_design_param2 = {'shin_length_1': 0.3,'shin_length_2': 0.3}
        # self.__origin_design_param3 = {'upper_arm_length_1': 0.2771,'upper_arm_length_2': 0.2771}
        # self.__origin_design_param4 = {'lower_arm_length_1': 0.2944,'lower_arm_length_2': 0.2944}
        # self.__origin_design_param5 = {'foot_length_1': 0.18,'foot_length_2': 0.18}
        self.__origin_design_params = {
            'thigh_lenth': 0.34,  # 大腿长 0.34
            'shin_lenth': 0.25,  # 小腿长 0.3
            'upper_arm_lenth': 0.21,  # 大臂长 0.2771
            'lower_arm_lenth': 0.22,  # 小臂长 0.2944
            'foot_lenth': 0.18,
            }  # 脚长   0.18
        # self.DNA_size       = decode_size * len(self.__origin_design_params)
        self.params_list = [
            self.__origin_design_params['thigh_lenth'],
            self.__origin_design_params['shin_lenth'],
            self.__origin_design_params['upper_arm_lenth'],
            self.__origin_design_params['lower_arm_lenth'],
            self.__origin_design_params['foot_lenth']
        ]
        self.train_data = np.array([
            self.__origin_design_params['thigh_lenth'],
            self.__origin_design_params['shin_lenth'],
            self.__origin_design_params['upper_arm_lenth'],
            self.__origin_design_params['lower_arm_lenth'],
            self.__origin_design_params['foot_lenth']
        ])

        
        self.n_envs = n_envs
        self.model = model
        
        self.terrain_type = terrain_type
        self.init_controller()
        self.gmm_initial_params = {
            'n_components': 1,
            'covariance_type': 'full',
            'random_state': 42
        }
        self.num_samples = 10
        self.num_epochs = num_epochs
        self.best_reward = list()
        self.fitness_data = list()
        self.last_best_design = list()
        self.sgd = None
        self.best_y = list()
        self.local_opt_flag = 0
    def init_controller(self, algo="sac"):
        # 初始化控制器
        RL_algo = {
            "sac": SAC,
            "td3": TD3,
            "ppo": PPO,
        }[algo]
        hyperparams = {
            "sac": dict(
                batch_size=256,
                gamma=0.98,
                policy_kwargs=dict(net_arch=[256, 256]),
                learning_starts=10000,
                buffer_size=int(5e2),
                tau=0.01,
                gradient_steps=4,
            ),
            "ppo": dict(
                batch_size=512,
                learning_rate=2.5e-4,
                policy_kwargs=dict(net_arch=({'pi': [128, 128]}, {'vf': [128, 128]})),
                gamma=0.99
            )
        }[algo]
        env_kwargs = {'terrain_type': self.terrain_type}
        self.envs = make_vec_env(env_id='HumanoidCustomEnv-v0', n_envs=self.n_envs, env_kwargs=env_kwargs)
        self.last_best_params = self.__origin_design_params
        # env = gym.make('HumanoidCustomEnv-v0', terrain_type='steps')
        # self.env = env
        # print('load from:')
        # save_path = 'best_model/5e6_steps_t5_cpu8_sac_HumanoidCustomEnv-v0.zip'
        # print(save_path)
        # self.model = SAC("MlpPolicy", env, verbose=1,  **hyperparams)
        # self.model.set_parameters(save_path)

        # 定义GMM的初始条件

    def robot_loss(self,params):
        # 更新机器人形态结构参数

        self.envs.update_xml_model(params)
        obs = self.envs.reset()
        episode = 0
        episode_rewards = []
        episode_reward = np.zeros(self.n_envs)
        while episode < self.n_envs:
            action, _ = self.model.predict(obs, )
            obs, rewards, dones, infos = self.envs.step(action)
            episode_reward += rewards
            for idx, done in enumerate(dones):
                if done:
                    print(idx)
                    episode += 1
                    episode_rewards.append(episode_reward[idx])
                    episode_reward[idx] = 0
        mean_reward = np.mean(episode_rewards)
        v_ave_list = list()
        for info in infos:  # 获取episode平均速度
            v_ave_list.append(info['ave_velocity'])
        v_ave = np.mean(v_ave_list)
        if self.terrain_type == 'default':
            min_reward = 550
        if self.terrain_type == 'steps':
            min_reward = 550
        if mean_reward < min_reward:
            v_ave = 0
        f = v_ave

        # if self.out_of_range(new_params, clip_range=0.1):
        #     # 如果参数更新幅度过大，惩罚20fitness
        #     f -= self.overchange_punish

        fitness = f

        self.best_reward.append(np.max(fitness))
        self.fitness_data.append(fitness)
        return fitness

    def get_fitness(self, params):
        pred = self.robot_loss(params)
        # return pred - np.min(pred)+1e-3
        return pred

    def new_design_params(self,p_thigh_lenth, p_shin_lenth, p_upper_arm_lenth, p_lower_arm_lenth, p_foot_lenth):

        params = {  'thigh_lenth':self.__origin_design_params['thigh_lenth']*p_thigh_lenth,
                    'shin_lenth':self.__origin_design_params['shin_lenth']*p_shin_lenth,
                    'upper_arm_lenth':self.__origin_design_params['upper_arm_lenth']*p_upper_arm_lenth,
                    'lower_arm_lenth':self.__origin_design_params['lower_arm_lenth']*p_lower_arm_lenth,
                    'foot_lenth':self.__origin_design_params['foot_lenth']*p_foot_lenth       }
        return params

    def evolve(self):
        # 进化N代
        global x_train, y_train
        optimized_params_coef = None
        
        for generation in range(self.num_epochs):
            
            if optimized_params_coef is None:
                
                generated_samples = self.get_gmm()
                x_train = generated_samples
                y_train = np.array([self.get_fitness(generated_samples)])
                self.last_best_design = generated_samples
                self.best_y = y_train
                optimized_params_coef = [1, 1, 1, 1, 1]
                
                
            if optimized_params_coef is not None:
                
                generated_samples = self.update_gmm(optimized_params_coef)
                x_train = generated_samples
                y_train = np.array([self.get_fitness(generated_samples)])  #负数越小越好
                print(y_train[0])
                print(max(self.best_y))
                if y_train[0] >= max(self.best_y):
                    optimized_params_coef=[]
                    self.best_y = y_train
                    for key in x_train:
                        temp_coef = x_train[key] / self.__origin_design_params[key]
                        optimized_params_coef.append(temp_coef)
                    self.last_best_design = x_train

                
                    



            
            # 更新机器人形态结构参数
        return optimized_params_coef

    def get_gmm(self):
        k = 5
        gmm = GaussianMixture(n_components=k, random_state=42)  # k是高斯分量的数量
        gmm.fit(self.train_data.reshape(-1, 1))
        generated_samples, labels = gmm.sample(100)
        reordered_samples = []
        sampled_indices = []  # 存储已选择的样本的索引
        selected_samples = []
        # 随机选择一个属于每个族的样本
        for i in range(k):
            indices = np.where(labels == i)[0]  # 获取属于当前族的样本索引
            chosen_index = random.choice(indices)  # 随机选择一个索引
            sampled_indices.append(chosen_index)
            selected_samples.append(generated_samples[chosen_index][0])
        reordered_samples = {
        'thigh_lenth': selected_samples[2],
        'shin_lenth': selected_samples[4],
        'upper_arm_lenth': selected_samples[3],
        'lower_arm_lenth': selected_samples[0],
        'foot_lenth': selected_samples[1]
        }
        return reordered_samples


    def update_gmm(self, optimized_params_coef):
        k = 5
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(self.train_data.reshape(-1, 1))  # 使用训练数据重新拟合GMM
        for component_idx in range(k):
            gmm.means_[component_idx] =gmm.means_[component_idx] * optimized_params_coef[component_idx]  # 更新均值
        generated_samples, labels = gmm.sample(100)
        reordered_samples = []
        sampled_indices = []  # 存储已选择的样本的索引
        selected_samples = []
        # 随机选择一个属于每个族的样本
        for i in range(k):
            indices = np.where(labels == i)[0]  # 获取属于当前族的样本索引
            chosen_index = random.choice(indices)  # 随机选择一个索引
            sampled_indices.append(chosen_index)
            selected_samples.append(generated_samples[chosen_index][0])
        reordered_samples = {
        'thigh_lenth': selected_samples[2],
        'shin_lenth': selected_samples[4],
        'upper_arm_lenth': selected_samples[3],
        'lower_arm_lenth': selected_samples[0],
        'foot_lenth': selected_samples[1]
        }
        return reordered_samples


    def test(self):
        pass



    def save_fig(self, fig_name: str = None):

        design_params = list(self.last_best_design.values())
        x = list(self.last_best_design.keys())

        plt.figure(figsize=(8, 8))
        plt.plot(x, design_params, 'ro-')
        plt.xlabel('Design Parameters')
        plt.ylabel('Values')
        plt.title('Design Parameters')
        plt.grid(True)
        
        if fig_name is None:
            fig_name = 'GMM_optim_result'
        
        plt.savefig('screenshot/' + fig_name + '.png', bbox_inches='tight')
        plt.show()