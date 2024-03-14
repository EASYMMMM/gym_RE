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
                 init_pos: float = np.pi,
                 w_e = 200,
                 w_q1 = 5,
                 w_q2 = 0.1,
                 w_r = 1, 
                 w_t = 30,
                 energy_obs = False,
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
        self.init_pos = init_pos # 初始角度
        self.energy_obs = energy_obs # 是否在观测空间中添加系统能量
        # reward系数
        self.w_e = w_e
        self.w_q1 = w_q1
        self.w_q2 = w_q2
        self.w_r = w_r
        self.w_t = w_t
        # 定义动作空间 (-3,3)
        #self.action_space = spaces.Box(
        #    low=np.array([-3.]),
        #    high=np.array([3.]),
        #    dtype=np.float32
        #)

        # 定义动作空间 [-3,0,3]
        self.action_space = spaces.Discrete(3)

        # 定义状态空间 (theta, theta_dot)
        if self.energy_obs:
            self.observation_space = spaces.Box(
                low=np.array([-np.pi, -15*np.pi, -np.pi, -200]),
                high=np.array([2*np.pi, 15*np.pi, np.pi, 200]),
            )
        else:
            self.observation_space = spaces.Box(
                low=np.array([-np.pi, -15*np.pi, -np.pi]),
                high=np.array([2*np.pi, 15*np.pi, np.pi]),
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
        return: state (observation)
        '''        
        m = self.m
        g = self.g
        l = self.l
        J = self.J
        b = self.b
        K = self.K
        R = self.R
        u = action*3 - 3
        dt = self.dt
        # 倒立摆动力学
        current_a = self.last_state[0]
        current_adot = self.last_state[1]        
        current_adotdot = (1/J)*(m*g*l*np.sin(current_a)-b*current_adot-K*K*current_adot/R+K*u/R)
        # 积分
        next_adot = current_adotdot*dt + current_adot 
        next_a = current_adot*dt + current_a
        if self.energy_obs:
            obs = np.array([next_a, next_adot , self.angle_to_target(next_a), self.system_energy],dtype=np.float32)
        else:
            obs = np.array([next_a, next_adot , self.angle_to_target(next_a)],dtype=np.float32)

        return obs

    def reward(self, state, action):
        '''
        计算reward
        '''
        current_a = state[0]
        current_adot = state[1]
        t = state[2]
        if current_a>np.pi:  # 角度处理
            current_a = 2*np.pi - current_a
        # 题目给定的奖励函数 
        R_1 = -self.w_q1*t*t - self.w_q2*current_adot*current_adot- self.w_r*action*action  
        # 达到目标位置的额外奖励
        if -0.2< current_a and current_a < 0.2 :
            R_2 = self.w_t  # target reward
        else:
            R_2 = 0
        # 能量奖励
        R_E = self.system_energy

        R = R_1 + R_2 + self.w_e*R_E
        return float(R)

    def reset(self):
        self.total_t = 0
        self.step_num = 0 # 计数器
        self.total_reward = 0
        self.success = False

        h = np.cos(self.init_pos)*self.l 
        v = 0 * self.l
        r_e = self.m*self.g*h + 0.5*self.m*v*v  
        if self.energy_obs:
            self.init_state = np.array([self.init_pos, 0 , self.angle_to_target(self.init_pos), r_e],dtype=np.float32)
        else:
            self.init_state = np.array([self.init_pos, 0 , self.angle_to_target(self.init_pos)],dtype=np.float32)
        self.last_state = self.init_state # 记录上一时刻的状态
        return self.init_state
    
    
    def done(self,state):  
        current_a = state[0]
        if current_a < -np.pi or current_a > 2*np.pi: # 限幅
            return True                      
        return False

    def step(self, action):
        state = self.__simulation(action)
        self.total_t += self.dt
        self.last_state = state # 记录当前状态
        reward = self.reward(state,action)
        self.total_reward += reward
        done = self.done(state)
        info = {"is_done":done}
        return state, reward, done, info
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        pass

    def angle_to_target(self,ang):
        # 旋转角度处理
        t = ang
        if ang>np.pi:  
            t = 2*np.pi - ang
        if ang<0 :
            t = -ang
        return t
    
    @property
    def system_energy(self):
        current_a = self.last_state[0]
        current_adot = self.last_state[1]
        h = np.cos(current_a)*self.l 
        v = current_adot * self.l
        r_e = self.m*self.g*h + 0.5*self.m*v*v  
        return r_e

''' 
使用训练好的模型，进行测试
'''
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def pendulum_animation(theta):
    # 倒立摆参数
    l = 0.42  # 摆长

    # 创建画布
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))
    ax.grid()

    # 初始化倒立摆
    line, = ax.plot([], [], 'o-', lw=2)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    pendulum_center, = ax.plot([], [], 'o', color='red')

    # 初始化函数
    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    # 更新函数
    def update(frame):
        x = [0, l * np.sin(frame)]
        y = [0, l * np.cos(frame)]
        line.set_data(x, y)
        time_text.set_text('Theta = %.2f' % np.degrees(frame))
        pendulum_center.set_data([0], [0])
        return line, time_text

    # 动画
    ani = animation.FuncAnimation(fig, update, frames=theta, init_func=init, blit=True)
    
    return ani

# 保存为GIF
def save_gif(ani, filename, fps=200):
    ani.save(filename, writer='imagemagick', fps=fps)



        


if __name__ == "__main__":
    # env = InvertedPendulum(render=True, init_pos=0)
    env = InvertedPendulum(render=True)
    from stable_baselines3.common.env_checker import check_env
    print('=='*20)
    print(env.observation_space)
    print(env.observation_space.sample())
    print(env.action_space.sample())
    print('=='*20)
    check_env(env)  #使用sb3自带的检查env函数来检查

    pend_a = list()
    i = 0
    frame_skip=10
    while i<2000:
        action = 2
        obs, reward, done, _ = env.step(action)
        
        print(f"state : {obs}, reward : {reward}")
        if i%frame_skip == 0:   # 降低绘图帧数
            pend_a.append(obs[0]) 
        
        i = i+1

    obs = env.reset()
    # 测试代码
    animation = pendulum_animation(pend_a)
    save_gif(animation, 'RL_train/pendulum_animation_test_3.gif')
    #plt.show()
