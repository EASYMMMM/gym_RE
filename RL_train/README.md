# 强化学习课程作业——倒立摆

孟令一 202328014628034

## 0. 环境

关键依赖：

```
  - numpy=1.23.5=py39h3b20f71_0
  - tensorboard=2.10.0=py39haa95532_0
  - gym==0.21.0
  - hydra-core==1.3.2
  - omegaconf==2.3.0
  - stable-baselines3==1.7.0
  - torch==1.12.1+cu116
```

使用`gym`构建强化学习环境。使用`hydra`配置实验参数。使用`stable-baselines3`提供的强化学习算法进行训练。

使用默认参数训练：

`python pendulum_train.py`

训练结果会保存在`runs\InvPend_xx-xx-xx-xx`文件夹中。其中，`xx-xx-xx-xx`为自动添加的时间戳。在该文件夹中，会包含后缀为`.zip`的模型结果，训练日志`tensorboard_log\`，参数配置`config.yaml`。

若要更改参数，可以通过hydra库在命令行中进行更改，如：

`python pendulum_train.py env.reward_scaling=False train.ent_coef=0.2`

检查训练结果：

`python pendulum_test.py model=InvPend_xx-xx-xx-xx`

其中，`InvPend_xx-xx-xx-xx`为训练自动生成的文件夹名称。运行该文件后，会在`runs\InvPend_xx-xx-xx-xx`下生成一个倒立摆运行结果的GIF图，以及一个运行结果曲线图。

## 1. 题目

![image-20240412215931534](C:\Users\孟一凌\AppData\Roaming\Typora\typora-user-images\image-20240412215931534.png)

## 2. 环境搭建

### 2.1 观测空间

题目要求中，观测空间定义为$\boldsymbol s = [\alpha, \dot{\alpha}]^T$。即倒立摆的角度和角速度。然而，通过经验得知，对于动力学系统，其加速度是非常重要的观测信息。于是在观测空间中额外加入加速度项$\ddot{a}$。

另外，倒立摆采用角度表示位置。而角度具有周期性，比如$\frac{2}{3}\pi$和$-\frac{4}{3}\pi$是相等的。在本实验中，$\frac{2}{3}\pi$和$-\frac{4}{3}\pi$应该做等价处理，以获得同样的奖励。然而，对于神经网络而言，这两个角度在数值上并不相等，可能会使得训练不稳定。若是采取处理，将角度限制在$[-\pi,\pi]$的区间内，当角度值跨越$\pi$时，会产生从$\pi$到$-\pi$的阶跃，这同样可能使得训练不稳定。因此采用笛卡尔空间系对角度进行表示，从而避免该问题。

于是，最终采用的观测空间为：
$$
\boldsymbol{s}=[\cos(\alpha),\sin(\alpha),\dot\alpha,\ddot\alpha]
$$

### 2.2 动作空间

采取连续的动作空间，即$a\in[-3,3]$。

### 2.3 仿真步

给定了系统的动力学方程：

<img src="C:\Users\孟一凌\AppData\Roaming\Typora\typora-user-images\image-20240414151732822.png" alt="image-20240414151732822" style="zoom: 67%;" />

在仿真中通过欧拉积分法实现：

```python
    def __simulation(self, action):     
        m = self.m
        g = self.g
        l = self.l
        J = self.J
        b = self.b
        K = self.K
        R = self.R
        dt = self.dt
        if self.discrete_action:
            u = action*3 - 3
        else:
            u = np.clip(3*action[0],-3,3)
        # 倒立摆动力学
        current_a = self.last_state[0]
        current_adot = self.last_state[1]  
        for i in range(self.frame_skip):      
            current_adotdot = (1/J)*(m*g*l*np.sin(current_a)-b*current_adot-K*K*current_adot/R+K*u/R)
            # 积分
            next_adot = current_adotdot*dt + current_adot 
            next_a = current_adot*dt + current_a
            current_a = next_a
            current_adot = next_adot
            self.episode_steps += 1

        state = np.array([next_a, next_adot , current_adotdot],dtype=np.float32)

        return state
```

需要说明了是，为了提高强化学习过程中的样本效率，每次仿真中采取10帧的frame skip。即使用相同的action连续进行10次dt=0.005的仿真，最终得到下一时刻的状态。

### 2.4 奖励函数

使用题目给定的奖励函数：

```python
    def reward(self, state, action):
        current_a = state[0]
        current_adot = state[1]
        if self.discrete_action:
            u = action*3 - 3
        else:
            u = np.clip(action[0],-3,3)
        # 题目给定的奖励函数 
        R = -self.w_q1*(self.angle_to_target(current_a)*self.angle_to_target(current_a)) - self.w_q2*current_adot*current_adot- self.w_r*u*u 
        if self.reward_scaling:
            R = (R+8)/8
        return float(R)
```

需要说明的是，为了提高训练过程中的稳定性，进行了reward scaling处理：$R=(R+8)/8$。此举能够降低奖励值的方差，提高训练稳定性。在计算角度项$\alpha$的奖励值时，对角度进行了限幅处理，全部重置到$[-\pi,\pi]$的区间内。

## 3. 训练

### 3.1 训练结果

使用stable-baselines3中的**PPO**算法进行训练。参数如下：

```yaml
# config.yaml
env:
    env_id: 'InvertedPendulumEnv-v0'
    model_name: 'InvPend'
    w_q1: 5
    w_q2: 0.1
    w_r: 1
    init_pos: 3.1415
    frame_skip: 10
    reward_scaling: True
    max_episode_steps: 2000
    discrete_action: False
train:
    n_timesteps: 3000000
    batch_size: 256
    gamma: 0.98
    learning_rate: 0.0003
    device: 'cuda:1'
    seed: 1
    use_sde: False
    ent_coef: 0.1
```

训练结果：

<img src="E:\CASIA\gym_RobotEvolution\RL_train\runs\InvPend_14-14-36-10\pendulum_animation.gif" alt="pendulum_animation" style="zoom:67%;" />

<img src="E:\CASIA\gym_RobotEvolution\RL_train\runs\InvPend_14-14-36-10\result_curve.png" alt="result_curve" style="zoom: 25%;" />

训练得到的倒立摆在左右摆动后，能够较快地升到最高处，然后保持稳定。

reward曲线如下：

<img src="C:\Users\孟一凌\AppData\Roaming\Typora\typora-user-images\image-20240414161313046.png" alt="image-20240414161313046" style="zoom:50%;" />

### 3.2 无reward scaling训练结果

仅取消reward scaling，其余不做改动的情况下，得到的结果如下：

<img src="E:\CASIA\gym_RobotEvolution\RL_train\runs\InvPend_14-15-27-40\pendulum_animation.gif" alt="pendulum_animation" style="zoom: 67%;" />

<img src="E:\CASIA\gym_RobotEvolution\RL_train\runs\InvPend_14-15-27-40\result_curve.png" alt="result_curve" style="zoom: 25%;" />

<img src="C:\Users\孟一凌\AppData\Roaming\Typora\typora-user-images\image-20240414161437093.png" alt="image-20240414161437093" style="zoom:50%;" />

可以看到，reward训练曲线呈下降趋势，无法有效学习。倒立摆运动策略则是在底部震荡的次优策略。