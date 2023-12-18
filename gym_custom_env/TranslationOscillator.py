'''
智能控制 2023秋 课程作业
设计具有旋转激励的平移振荡器系统的控制器

定义平移振荡器的强化学习环境
'''


import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class TranslationOscillator(gym.Env):
    def __init__(self, render : bool = False, 
                 random_seed = None, 
                 epsilon : float = 0.1,
                 simulation_dt : float = 0.02,
                 reward_weight : list = [2,0.2,1,0.1],
                 frame_skip :int = 5,
                 random_init  = False,  # 随机初始状态
                 suqare_reward = False, # 计算reward时，平方处理
                 acc_state = False,     # 观测状态中是否添加两个加速度
                 stable_reward = 0,     # 稳定额外奖励
                 stable_limit = 0.1,    # 稳定阈值  
                 init_state = [0,0,0,0],# 初始状态 （仅当random_init  = False）
                 ):
        self._render = render
        # 定义动作空间
        self.action_space = spaces.Box(
            low=np.array([-5.]),
            high=np.array([5.]),
            dtype=np.float32
        )

        # 定义状态空间 (x1, x2, x3, x4)
        if acc_state:
            self.observation_space = spaces.Box(
                low=np.array([-2.,-2.,-np.pi,-2.,-2,-2]),
                high=np.array([2., 2., np.pi, 2., 2, 2]),
            )
        else:
            self.observation_space = spaces.Box(
                low=np.array([-2.,-2.,-np.pi,-2.]),
                high=np.array([2., 2., np.pi, 2.]),
            )

        if random_seed != None:
            self.seed(random_seed)
        self.epsilon = epsilon # 耦合系数
        self.dt = simulation_dt # 仿真时间步长
        self.reward_weight = reward_weight # reward权重
        self.frame_skip = frame_skip # 训练跳过的步数
        self.random_init = random_init # 随机初始状态
        self.suqare_reward = suqare_reward # 平方reward
        self.acc_state = acc_state 
        self.last_xdd = 0 # 记录振荡平台的加速度
        self.last_qdd = 0 # 记录小球的加速度
        self.stable_reward = stable_reward # 额外奖励
        self.stable_limit = stable_limit
        self.init_state = init_state
        self.reset()
        
    
    
    def __simulation(self, action):
        '''
        根据给定的公式 x' = f(x) + g(x)N + q(x)f
        进行一步仿真计算
        计算出下一时刻的状态
        '''
        
        epsilon = self.epsilon
        current_x =  self.last_state[0:4]  # 当前状态
        for i in range(self.frame_skip):
            x = current_x
            F = np.array([0.0,0.0,0.0,0.0])
            G = np.array([0.0,0.0,0.0,0.0])
            Q = np.array([0.0,0.0,0.0,0.0])
            F[0] = x[1]
            F[1] = (-x[0]+epsilon*x[3]*x[3]*np.sin(x[2]))/(1-epsilon*epsilon*np.cos(x[2])*np.cos(x[2]))
            F[2] = x[3]
            F[3] = (epsilon*np.cos(x[2])*(x[0]-epsilon*x[3]*x[3]*np.sin(x[2])))/(1-epsilon*epsilon*np.cos(x[2])*np.cos(x[2]))   
            G[0] = 0 
            G[1] = (-epsilon*np.cos(x[2]))/(1-epsilon*epsilon*np.cos(x[2])*np.cos(x[2]))
            G[2] = 0 
            G[3] = (1)/(1-epsilon*epsilon*np.cos(x[2])*np.cos(x[2]))
            Q[0] = 0
            Q[1] = (1)/(1-epsilon*epsilon*np.cos(x[2])*np.cos(x[2]))
            Q[2] = 0
            Q[3] = (-epsilon*np.cos(x[2]))/(1-epsilon*epsilon*np.cos(x[2])*np.cos(x[2]))
            f = np.array(0.1*np.sin(self.total_t*np.pi))
            N = np.array(action)
            x_dot = F + G*N + Q*f
            next_x = x + x_dot*self.dt
            if self.acc_state:
                self.last_xdd = x_dot[1]
                self.last_qdd = x_dot[3]
            current_x = next_x
        return np.array(next_x,dtype=np.float32)

    def reward(self, x):
        '''
        计算reward
        对每一个状态量, s将其接近0的程度归一化到[-0.5,0.5]
        '''
        w = self.reward_weight
        r = 0
        #scale = 1/self.stable_limit
        scale = 1
        a = x[0]*scale
        b = x[2]*scale
        if self.suqare_reward:
            r += -w[0]*np.abs(a)*np.abs(a)/2/2 
            r += -w[1]*np.abs(x[1])*np.abs(x[1])/2/2
            r += -w[2]*np.abs(b)*np.abs(b)/np.pi/np.pi
            r += -w[3]*np.abs(x[3])*np.abs(x[3])/2/2 
        else:
            r += -w[0]*np.abs(a)/2 
            r += -w[1]*np.abs(x[1])/2 
            r += -w[2]*np.abs(b)/np.pi 
            r += -w[3]*np.abs(x[3])/2 

        stable_limit = self.stable_limit
        # 鼓励稳定在原点附近
        if  (x[0]<stable_limit and x[0]>-stable_limit) :
            r += self.stable_reward
            if  (x[2]<stable_limit and x[2]>-stable_limit) :
                r += self.stable_reward

        # 每步仿真奖励值[-2.2,0]，每秒（50Hz）最大得到100奖励值\

        # REWARD 取状态量中偏离目标状态最大的，做惩罚
        # s = np.abs(x)/np.array([10,10,np.pi,10]) 
        # r =  - s[np.where(s==np.max(s))]
        #if self.total_t > 20:
        #    # 运行时长超过20秒后，施加惩罚
        #    r += -(self.total_t-20)  
        return float(r)

    def reset(self, init_state :np.ndarray= None):
        if self.random_init:
            # 在[+1, 0, +1, 0]的范围内随机初始化
            r = np.random.rand()
            # if r > 0.5:  # 一半初始状态，一半目标状态
            #     init_state = np.array([0.25,0,0,0])
            # else:
            #     init_state = np.array([0,0,0,0])
            init_state = [r*self.init_state[0], 0, 0 ,0]
            if r < 0.2:
                init_state = [0,0,0,0]
        else:
            init_state = np.array(self.init_state)
        self.total_t = 0
        self.step_num = 0 # 计数器
        self.total_reward = 0
        self.success = False
        if self.acc_state:
            state = np.append(init_state,[0,0])
        else:
            state = init_state
        self.last_state = np.array(state) # 记录上一时刻的状态
        return np.array(state,dtype=np.float32)
    
    @property
    def done(self):
        d = False
        if self.total_reward < -3000: # 当负值奖励积累的太大时，提前终止训练
            d = True
        if self.total_reward > 1500: # 总奖励达到1500，认为训练成功
            d = True
            self.success = True
        # 偏差较大时提前终止
        if self.last_state[0] > 4 or self.last_state[0]<-4: 
            d = True                          
        return d

    def step(self, action):
        state = self.__simulation(action)
        self.total_t += self.dt*self.frame_skip
        self.last_state = state # 记录当前状态
        reward = self.reward(state)
        self.total_reward += reward
        done = self.done
        info = {"is_success":self.success}
        
        scale = np.array([5,5,np.pi,5])
        #state = state / scale
        if self.acc_state:
            acc = np.array([self.last_xdd,self.last_qdd])
            state = np.concatenate((state,acc) )
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
    #plt.plot(all_x2, label = 'x2')
    plt.plot(all_x3, label = 'x3')
    #plt.plot(all_x4, label = 'x4')
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
    env = TranslationOscillator(render=True)
    from stable_baselines3.common.env_checker import check_env
    print('=='*20)
    print(env.observation_space)
    print(env.observation_space.sample())
    print('=='*20)
    check_env(env)  #使用sb3自带的检查env函数来检查

    obs = env.reset()
    while True:
        action = np.random.uniform(-10, 10, size=(1,))
        obs, reward, done, _ = env.step(action)
        if done:
            break

        print(f"state : {obs}, reward : {reward}")
