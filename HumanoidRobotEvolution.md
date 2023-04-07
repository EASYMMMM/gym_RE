# Humanoid Robot Evolution

## 零、题目

**Co-optimization of Morphology and Controller of Complex Robot's Based on DRL**

**基于深度强化学习的复杂机器人形态-控制联合优化**

## 一、 背景

### 1. 机器人形态优化文献综述

- **Data-efficient Co-Adaptation of Morphology and Behaviour with Deep Reinforcement Learning**

  <img src="C:\Users\孟一凌\AppData\Roaming\Typora\typora-user-images\image-20230405170205407.png" alt="image-20230405170205407" style="zoom:50%;" />

  基于强化学习，对形态和控制器联合优化。

  框架构建：

  - 构建MDP模型，将设计参数$\xi$添加到MDP中。策略$\pi(s,\xi)$同样依赖于设计参数。

  ![image-20230405202217974](C:\Users\孟一凌\AppData\Roaming\Typora\typora-user-images\image-20230405202217974.png)

  **整体算法结构：**

  ---

  1.  初始化Replay Buffer，和Actor，Critic网络

  2. 生成一个初始设计$\xi$。
  3. 使用全局控制网络的参数对个体控制网络进行赋值。
  4. 训练设计$\xi$。保存马尔可夫链的数据至全局Buffer和个体Buffer。用得到的数据分别训练全局控制网络（$\pi_{Pop.}, Q_{Pop}$，实际上即Actor网络和Critic网络）和针对本次设计的个体控制网络（$\pi_{Ind.}, Q_{Ind}$）

  3. 保存该次训练的起始状态$S_O$。

  4. 根据当前迭代数决定是进行Exploitation还是Exploration:
     - Exploitation：根据得到的全部$S_O$数据来计算待优化的目标函数：$max_{\xi}\frac{1}{n}\sum_s\in s_{batch} {Q_{Pop.}(s,\pi_{Pop.}(s,\xi),\xi)}$
     - Exploration：根据探索策略选择新的设计$\xi$。
  5. 返回步骤2

  ---

  **为何能降低运算时间**：将单个设计的训练数据同时用于训练全局训练网络。此全局训练网络可应用于后续的新的设计。由此，每个新的设计无需从零开始学习。

### 2. 为什么进行机器人的控制器-形态联合优化？

机器人的形态决定其功能。不同的形态又需要不同的控制器。在针对特定任务设计并优化机器人形态时，需要始终评估不同形态的性能。为众多形态分别设计控制器并进行实验是非常费时费力的。希望能够在仿真中构建控制器-形态联合优化的框架，减少优化时长，并提升最终效能。

## 二、 机器人控制器

### 1. 强化学习控制器

使用强化学习训练一个策略网络，作为机器人（agent）的控制器。可选择的网络有两个：SAC和PPO。

PPO的基本思想跟PG(Policy Gradient)算法一致，直接根据策略的收益好坏来调整策略。PPO是一种on-policy的算法，这类算法在训练时需要保证生成训练数据的policy与当前训练的policy一致，对于过往policy生成的数据难以再利用，所以在sample efficiency这条衡量强化学习算法的重要标准上难以取得优秀的表现。

在实际应用中，PPO算法总是可以获得一条稳定上升的episode-reward曲线，但曲线的斜率很低，如果要达到足够高的reward需要非常多的timestep（对于我们的任务可能是e7到e8的量级）。

SAC是一个off-policy，actor-critic算法。与其他RL算法最为不同的地方在于，SAC在优化策略以获取更高累计收益的同时，也会最大化策略的熵。SAC中的熵（entropy）可以理解为混乱度，无序度，随机程度，熵越高就表示越混乱，其包含的信息量就更多。假设有一枚硬币，不论怎么投都是正面朝上，则硬币的正反这一变量的熵就低，如果正反出现的概率都为0.5，则该变量的熵相对更高。将熵引入RL算法的好处为，可以让策略（policy）尽可能随机，agent可以更充分地探索状态空间S，避免策略早早地落入局部最优点（local optimum），并且可以探索到多个可行方案来完成指定任务，提高抗干扰能力。

###  2. Reward

强化学习中，一般鼓励使用宽泛的reward function。

若想让agent做出某些特定的动作，需要根据任务要求来精细地设计reward function。

#### 2.1 楼梯地形的Reward Function

目前的reward function：
$$
R = w_{forward}r_{foward} + w_{healthy}r_{healthy} + w_{stand}r_{stand} - w_{control}c_{control} - w_{contact}c_{contact}
$$
其中，前进奖励包括了前进距离和前进速度：
$$
r_{forward} = w_{speed}v_x + w_{distace}x
$$
存活奖励为布尔值。当机器人被判断为存活时，始终能够得到该奖励。

站立奖励计算机器人躯干坐标系的z轴与世界坐标系的z轴的重合程度。
$$
\vec{z} = [0 , 0,1] \\
r_{stand} = \vec{f_z} \cdot  \vec{z}
$$

目前的reward function是连续的奖励函数，能够鼓励agent在楼梯地形中沿着楼梯向上前进。但同样也是由于奖励函数是连续的，且并没有惩罚项，导致agent会通过一些“偷懒”的方式完成任务。比如冲上台阶来获得前进奖励，或原地不动获得生存奖励。针对这些问题，已经减少了生成奖励的权重，增加了位移奖励的权重，以此鼓励agent能够走上台阶。

受

#### 2.2 梯子地形的Reward Function

梯子地形的reward function目前基本的形式与楼梯地形相同：
$$
R = w_{forward}r_{foward} + w_{healthy}r_{healthy} + w_{stand}r_{stand} - w_{control}c_{control} - w_{contact}c_{contact}
$$
其中的前进奖励项更改为向上高度的奖励：
$$
r_{forward} = w_{speed}v_x + w_{height}z
$$
作为调整，调低了前进速度权重$w_{speed}$，调高了高度奖励$w_{height}$，调低了生存奖励$w_{healthy}$，调低了直立奖励$w_{stand}$。目的是进一步鼓励机器人沿着梯子上升。

问题也非常明显。相比于走楼梯，爬梯子是一系列更为复杂的动作，需要全身协调配合。因此，尽管有着向上高度的奖励函数来引导它向上，但是agent并不知道如何向上。

<img src="C:\Users\孟一凌\AppData\Roaming\Typora\typora-user-images\image-20230404152020410.png" alt="image-20230404152020410" style="zoom:50%;" />

为了更好的引导agent爬上梯子，考虑进一步添加离散奖励，奖励agent与梯子接触，来对agent进行阶段性的引导。这样的reward能够帮助agent更好地探索空间，并作为分目标，帮助其训练。

初步考虑为：

- 当agent的手或者脚与梯子接触时，给予高额reward。
- 每个手/脚接触每层梯子的reward仅给一次，防止agent将手放在梯子上不动。
- 越高处的梯子给的reward越高。

#### 2.3 离散Reward Function在本任务中的具体实现

在Mujoco中，每次`step`迭代都会计算一次接触，接触数据以`Mjcontact`数组的形式储存在`Pymjdata`中。

Github上关于mjcontact的issue： https://github.com/openai/mujoco-py/issues/725

可以通过`sim.data.ncon` 来查询几何体碰撞对的个数，通过` sim.data.contact`来访问`Mjdata`中的`contact`数据，这是一个结构体数组。该结构体定义如下。

```C++
struct _mjContact                   // result of collision detection functions
{
    // contact parameters set by geom-specific collision detector
    mjtNum dist;                    // distance between nearest points; neg: penetration
    mjtNum pos[3];                  // position of contact point: midpoint between geoms
    mjtNum frame[9];                // normal is in [0-2]

    // contact parameters set by mj_collideGeoms
    mjtNum includemargin;           // include if dist<includemargin=margin-gap
    mjtNum friction[5];             // tangent1, 2, spin, roll1, 2
    mjtNum solref[mjNREF];          // constraint solver reference
    mjtNum solimp[mjNIMP];          // constraint solver impedance

    // internal storage used by solver
    mjtNum mu;                      // friction of regularized cone, set by mj_makeR
    mjtNum H[36];                   // cone Hessian, set by mj_updateConstraint

    // contact descriptors set by mj_collideGeoms
    int dim;                        // contact space dimensionality: 1, 3, 4 or 6
    int geom1;                      // id of geom 1
    int geom2;                      // id of geom 2

    // flag set by mj_fuseContact or mj_instantianteEquality
    int exclude;                    // 0: include, 1: in gap, 2: fused, 3: equality

    // address computed by mj_instantiateContact
    int efc_address;                // address in efc; -1: not included, -2-i: distance constraint i
};
typedef struct _mjContact mjContact;
```

每次读取` sim.data.contact`，会得到一个不定长的数组，其中包括了所有的集合体碰撞对，数组中的每一个`mjContact`元素包含了一对碰撞体的信息。对于本任务而言，有用的参数即`geom1`，`geom2`。这两个参数返回了碰撞对中两个geom几何体的序号。注意，是序号，且序号顺序由XML文档中，在`wolrdbody`下定义的`geom`顺序决定。

在mujoco和mujocopy的文档中暂时未找到由geom序号得到geom名称的函数。仅有函数` sim.model.geom_name2id("xxx")`。这个函数会返回名称为'xxx'的几何体的序号。当然，需要在XML中对几何体进行命名。因此，在构建mujoco仿真环境时，先对XML中的全部几何体进行命名，然后根据` sim.model.geom_name2id("xxx")`函数构建一个键值对为`{name:id}`的字典。此后便可通过在XML中给几何体定义的名称来查询该几何体。

以下是一个对上述函数的例程。

```python
# ------- 来自于mujoco150在win+py3.9下的矫情的要求 --------
# 手动添加mujoco路径
import os
from getpass import getuser
import time
user_id = getuser()
os.add_dll_directory(f"C://Users//{user_id}//.mujoco//mujoco200//bin")
os.add_dll_directory(f"C://Users//{user_id}//.mujoco//mujoco-py-2.0.2.0//mujoco_py")
# -------------------------------------------------------
import mujoco_py
import numpy as np
from mujoco_py.generated import const

# 加载 XML 文件
model = mujoco_py.load_model_from_path("humanoid.xml")

# 创建仿真环境和渲染器
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

# 设置仿真时间步长和仿真时长
dt = 0.01
timesteps = 500000
viewer.render()

# 运行仿真并在每个时间步骤中进行渲染
for i in range(timesteps):
    sim.step()
    viewer.render()

    # 读取mujoco碰撞参数
    contact = list(sim.data.contact) # 读取一个元素为mjContact的结构体数组
    print('==================================')
    print('geom number: ', sim.model.ngeom)
    print('number of detected contacts:',sim.data.ncon)
    print('geom name floor id:' , sim.model.geom_name2id("floor")) # 查询名字为‘floor’的几何体的id
    print('geom name lwaist id:' , sim.model.geom_name2id("lwaist")) # 查询名字为‘lwaist’的几何体的id
    # 打印第1个碰撞对的两个几何体的id
    print('geom1 id:',contact[1].geom1,' geom1 name:',sim.model.name_geomadr[contact[0].geom1])
    print('geom2 id:',contact[1].geom2,' geom2 name:',sim.model.name_geomadr[contact[0].geom2])
    print('  ')

# 关闭仿真环境和渲染器
viewer.close()
sim.close()


```

奖励函数的输入：

- agent的状态（`sim.data.contact`）

奖励函数需要完成：

- 读取contact信息。
- 扫描contact数组，寻找其中是否有‘手-梯子’，‘脚-梯子’的碰撞对。注意geom1既可能是梯子也可能是手。
- 如果有，将这一对碰撞对保存下来。若该碰撞对已存在，则跳过，不获得奖励函数。
- 根据梯子的阶数，赋予奖励值。梯子越高，奖励值越高。

奖励函数的输出:

- Reward值



# 文档留存

## 开题报告

### 选题依据

Robots are often required to work in complex terrains and environments. The robots that work in different environments requires the corresponding structure, including the actuators, the morphology, the size, etc. For example, a robot driving in rugged mountainous terrain needs more flexible joints to help it to cross obstacles and climb. Thus, good structural design is critical to a robot's ability to perform tasks in complex environments. On the other side, the manual design for robot structure is time consuming and requires necessary expertise.

In this research, we aim to develop a robot structure evolution mode to accelerate the robot structure design for a specific task. The key issue is to find the optimal solution from the enormous design space. To address the issue, we aim to design a heurist-based search method combined with the generative grammar model and the reinforcement learning control. The generative rule and the robot dynamic model are evaluated in the robot simulation platform. With generations of evolvement, the robot is able adapt to different terrains in the simulation environment and accomplish motion tasks such as moving smoothly forward and crossing obstacles.

The field of "robot evolution" has received a lot of attention in recent years. The main research directions are the optimization of robot morphology and the joint optimization of robot morphology and controller. However, a single controller is not able to adapt to robots with different design parameters, and may lead to poor optimization results. Therefore, the joint optimization of robot morphology and controller is becoming a mainstream research direction. Optimization methods fall into two main categories: nonlinear programming and evolutionary algorithms. Nonlinear programming are more effective in optimization, but require explicit dynamical models and physical parameters of the system. When the robot and the environmental system become complex, modeling can be very difficult. Evolutionary algorithms, on the other hand, do not require any assumptions about the dynamics of the system or the objective function. It can be applied to a large number of problems, but the evolutionary efficiency is low and the optimization is relatively ineffective.

Another emerging approach is the neural network-based optimization method. Thanks to the rapid development of deep learning, especially deep reinforcement learning, neural approaches have gained more attention. The neural approaches can give the optimal controller of the design more efficiently, thus improving the efficiency of co-optimization.

### 研究目标和内容

Research objectives.

1. Propose a co-optimization algorithm for morphology and controller of humanoid robot.

2. Complete the co-optimization of morphology and controller for simple humanoid robot.

3. Complete the co-optimization design of the existing humanoid robot.

Main contents.

1. Establish a simulation platform to be able to evaluate different humanoid robots under different terrains (tasks).

2. Complete controller optimization for any given humanoid robot based on reinforcement learning.

3. Optimize the morphology of the robot using heuristic methods.

4. Further extension of the joint optimization algorithm to existing humanoid robots.

Key Questions.

1. How to improve the efficiency of reinforcement learning in the each iteration of co-optimization.

2. How to find the optimal solution in a large design space.

### 研究方案

1. Optimize the robot's controller using deep reinforcement learning.

2. Optimizing the morphology of the robot using heuristic algorithms.

3. combine the two optimization algorithms.

4. Perform simulation experiments.

**Feasibility analysis:**

Compared to the already proposed co-optimization algorithm, the biggest change in this algorithm is to change the agent from a robot with a simple morphological structure to a humanoid robot. It requires better efficiency of the controller optimization algorithm. At the same time, the design space of the humanoid robot is large and discrete, so the design of the heuristic optimization algorithm will be more difficult.

### 创新点及预期研究成果

Innovation points.

1. Humanoid robot as the object of robot morphology evolution.

2. A co-optimization framework based on deep reinforcement learning, heuristic search algorithms.

 

Expected outcomes.

1. A simulation platform that can optimize a variety of humanoid robots for different tasks.

2. One sci/ei paper published with the results

