'''
强化学习 2024春 课程作业
倒立摆

定义倒立摆的强化学习环境
'''


import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class InvertedPendulum(gym.Env):
    def __init__(self, render : bool = False, 
                 random_seed = None, 
                 simulation_dt : float = 0.005,
                 frame_skip :int = 5,
                 ):
        self._render = render
        # 物理参数
        self.m = 0.055 # mass
        self.g = 9.81  # gravity
        self.l = 0.042 # arm lenth
        self.J = 1.91e-4 # Inertia
        self.b = 3e-6  # damping
        self.K = 0.0536 # torque constant
        self.R = 9.5   # resistance      
        
        # 定义动作空间 (-3,3)
        self.action_space = spaces.Box(
            low=np.array([-3.]),
            high=np.array([3.]),
            dtype=np.float32
        )

        # 定义状态空间 (theta, theta_dot)
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -15*np.pi]),
            high=np.array([np.pi, 15*np.pi]),
        )

        if random_seed != None:
            self.seed(random_seed)
        self.dt = simulation_dt # 仿真时间步长
        self.frame_skip = frame_skip # 训练跳过的步数 
        self.reset()
        
    
    
    def __simulation(self, action):
        '''
        根据给定的公式 x' = f(x) + g(x)N + q(x)f
        进行一步仿真计算
        计算出下一时刻的状态
        '''        
        m = self.m
        g = self.g
        l = self.l
        J = self.J
        b = self.b
        K = self.K
        R = self.R
        u = action
        dt = self.dt

        current_a, current_adot =  self.last_state  # 当前状态
        # 倒立摆动力学
        current_adotdot = 1/J*(m*g*l*np.sin(current_a)-b*current_adot-K*K*current_adot/R+K*u/R)
        # 积分
        next_adot = current_adotdot*dt + current_adot 
        next_a = current_adot*dt + current_a
        return np.array([next_a,next_adot],dtype=np.float32)

    def reward(self, state, action):
        '''
        计算reward
        '''
        current_a, current_adot = state
        R = -5*current_a*current_a -0.1*current_adot*current_adot- action*action
        return float(R)

    def reset(self):
        self.total_t = 0
        self.step_num = 0 # 计数器
        self.total_reward = 0
        self.success = False
        self.init_state = np.array([-np.pi,0],dtype=np.float32)

        self.last_state = self.init_state # 记录上一时刻的状态
        return self.init_state
    
    @property
    def done(self):                         
        return False

    def step(self, action):
        state = self.__simulation(action)
        self.total_t += self.dt
        self.last_state = state # 记录当前状态
        reward = self.reward(state,action)
        self.total_reward += reward
        done = self.done
        info = {"is_done":done}
        return state, reward, done, info
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        pass
    

''' 
使用训练好的模型，进行测试
'''
def play( env, model, init_state: np.ndarray = None ,csv_path = 'TORA.csv'):
    import pandas as pd
    all_x1 = list()
    all_x2 = list()
    all_x3 = list()
    all_x4 = list()
    output = list()
    episode_rewards, episode_lengths, = [], []
    
    
    obs = env.reset(init_state = init_state)   
    all_x1.append(obs[0])
    all_x2.append(obs[1])
    all_x3.append(obs[2])
    all_x4.append(obs[3])
    done = False
    episode_reward = 0.0
    episode_length = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        print(reward)
        episode_reward += reward
        episode_length += 1
        output.append(action)
        all_x1.append(obs[0])
        all_x2.append(obs[1])
        all_x3.append(obs[2])
        all_x4.append(obs[3])
    output.append(action)
    is_success       = info['is_success']
    episode_rewards.append(episode_reward)
    episode_lengths.append(episode_length)
    print('************************')    
    print(
        f"Episode {len(episode_rewards)} reward={episode_reward}, length={episode_length}"
    )
    print('success:',is_success)
    print('************************')    
    plt.figure(_)
    plt.plot(all_x1, label = 'x1')
    plt.plot(all_x2, label = 'x2')
    plt.plot(all_x3, label = 'x3')
    plt.plot(all_x4, label = 'x4')
    #plt.plot(output, linestyle = '-',label = 'output')
    plt.legend()
    plt.grid(True,linestyle = '--')
    plt.show()

    data ={'x1':all_x1,
           'x2':all_x2,
           'x3':all_x3,
           'x4':all_x4,
           'output':output}
    df = pd.DataFrame(data)
    filename = csv_path
    df.to_csv(filename,index=False)
    print(f'Write data to {filename}.')


        


if __name__ == "__main__":
    env = InvertedPendulum(render=True)
    from stable_baselines3.common.env_checker import check_env
    print('=='*20)
    print(env.observation_space)
    print(env.observation_space.sample())
    print('=='*20)
    check_env(env)  #使用sb3自带的检查env函数来检查

    obs = env.reset()
    while True:
        action = np.random.uniform(-3, 3, size=(1,))
        obs, reward, done, _ = env.step(action)
        if done:
            break

        print(f"state : {obs}, reward : {reward}")
