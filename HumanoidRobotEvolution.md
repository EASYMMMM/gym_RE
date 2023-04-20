# 1Humanoid Robot Evolution

## 零、题目

原题目： Robot structure evolution based on heuristic search

**Co-optimization of Morphology and Controller of Humanoid Robot Based on DRL**

**基于深度强化学习的复杂机器人形态-控制联合优化**

## 一、 背景

### 1. 机器人形态优化文献综述

#### 1.1 形态优化/联合优化

---

- **Data-efficient Co-Adaptation of Morphology and Behavior with Deep Reinforcement Learning**

  - 基于强化学习，对形态和控制器联合优化。

    框架构建：

    ​		构建MDP模型，将设计参数$\xi$添加到MDP中。策略$\pi(s,\xi)$同样依赖于设计参数。

    <img src="C:\Users\孟一凌\AppData\Roaming\Typora\typora-user-images\image-20230405170205407.png" alt="image-20230405170205407" style="zoom:50%;" />

    <img src="C:\Users\孟一凌\AppData\Roaming\Typora\typora-user-images\image-20230405202217974.png" alt="image-20230405202217974" style="zoom: 67%;" />

  - ​	**整体算法结构：**

    1.  初始化Replay Buffer，和Actor，Critic网络
    2.  生成一个初始设计$\xi$。
    3.  使用全局控制网络的参数对个体控制网络进行赋值。
    4.  训练设计$\xi$。保存马尔可夫链的数据至全局Buffer和个体Buffer。用得到的数据分别训练全局控制网络（$\pi_{Pop.}, Q_{Pop}$，实际上即Actor网络和Critic网络）和针对本次设计的个体控制网络（$\pi_{Ind.}, Q_{Ind}$）
    5.  保存该次训练的起始状态$S_O$。
    6.  根据当前迭代数决定是进行Exploitation还是Exploration:
        - Exploitation：根据得到的全部$S_O$数据来计算待优化的目标函数：$max_{\xi}\frac{1}{n}\sum_s\in s_{batch} {Q_{Pop.}(s,\pi_{Pop.}(s,\xi),\xi)}$
        - Exploration：根据探索策略选择新的设计$\xi$。
    7.  返回步骤2

  - **为何能降低运算时间**：将单个设计的训练数据同时用于训练全局训练网络。此全局训练网络可应用于后续的新 的设计。由此，每个新的设计无需从零开始学习。

- **Application of reinforcement learning in biped robot gait controlling**  

  - 考虑到当 action 维度增高时，强化学习的收敛性与训练效率会降低 ，因此将机器人上半身关节固定 ，保留髋关节 3 个自由度、膝关节 1 个自由度与踝关节 2 个自由度 ，双腿共计 12 个自由度 ，对应action的12个维度。
  - 因为在实际行走中，机器人的运动应该为稳定持久 ，不随行走距离增大而发生改变的，因此在观测中，不应包括机器人在仿真
    世界中的前进距离 ，否则待训练智能体的决策将受到此信息的干扰，当机器人其他信息相同，只有前进距离不同时 ，会出现不同的动作 ，影响持久性。  
  
- **Reinforcement Learning for Improving Agent Design**

  <img src="C:\Users\孟一凌\AppData\Roaming\Typora\typora-user-images\image-20230413165832685.png" alt="image-20230413165832685" style="zoom: 33%;" />

  使用Ant模型进行设计优化。

  将环境相关的设计参数设置为可学习的参数，与控制器一起学习。

  - 保持所有材料的体积质量密度以及电机关节的参数与原始环境相同，并允许36个参数(每个leg part 3个参数，每个leg part 3个，共4个leg)被学习。特别是，我们允许每个部分缩放到其原始值的$±75\%$范围。这使我们能够保留每个部分的标志和方向，以保留设计的原始预期结构。
  - In principle, we could allow an agent to augment its body during a rollout to obtain a dense
    reward signal, but we find this impractical for realistic problems. **Future work may look at separating the learning from dense rewards and sparse rewards into an inner loop and outer loop**, and also examine differences in performance and behaviors in structures learned with various different RL algorithms.
  - 来自刘华平的评价：文献 [99] 针对越障问题利用强化学习实现形态与控制策略的联合学习.   

- **Jointly learning  to  construct  and  control  agents  using  deep  reinforcement learning**

  - 来自刘华平的评论：将设计参数与控制参数一起用近端策略优化方法（PPO）联合计算. 由于形态搜索空间过大, 且形态与控制的搜索难以解耦, 学习收敛非常困难. 因此作者对形态搜索的空间做了约束, 仅能针对指定的形态优化机器人组件的参数, 而没有优化机器人的结构.   

- **基于形态的具身智能研究：历史回顾与前沿进展**

  - 尽管形态与控制应该联合协同优化, 但二者其实是在不同尺度上的优化. 以生物为例, 形态的变化 (包括结构与参数) 更类似一种进化过程, 即在长期的环境适应过程中通过进化过程来优化自身的结构与参数; 而控制器的设计过程更类似于后天的学习过程, 即在确定形态后在自己的生命期内通过学习努力达到运动能力的边界. 因此不难看出, 一个很自然的想法是利用进化优化方法实现形态结构与参数的寻优, 而利用强化学习策略实现控制结构与参数的优化. 二者嵌套在两个回路中, 其中进化优化方法为外部循环, 而强化学习为内部循环.   

#### 1.2 Reward shaping

强化学习中，一般鼓励使用宽泛的reward function。

若想让agent做出某些特定的动作，需要根据任务要求来精细地设计reward function。

**什么是Reward Shaping？**

> 当我们使用强化学习算法来解决一个特定的任务时，我们需要定义一个目标，也称为“奖励函数”或“回报函数”，来指导代理在环境中采取哪些行动。在某些情况下，这个奖励函数可能很难直接设计，或者可能存在一些不希望代理学到的副作用。这时候就需要使用奖励塑形（Reward shaping）技术来调整奖励函数，以更好地指导代理的学习。
>
> 奖励塑形的基本思想是通过修改奖励函数来改变代理在环境中的行为。**这种修改通常是基于对环境的先验知识或经验的利用，可以帮助代理更快地学习到好的策略，或避免不必要的代价。**例如，在一个围棋游戏中，如果我们只为代理在胜利时提供正奖励，那么代理可能会倾向于采取过于冒险的策略，以尽可能早地获得胜利。但如果我们为代理在每个步骤上的棋子数量提供适当的奖励，那么代理可能会更倾向于保持自己的棋子数量，并采取更稳健的策略。
>
> 总之，奖励塑形是一种非常有用的技术，可以帮助我们更好地指导代理在环境中学习到好的策略。
>
> -- 知乎

> 一个不合适的reward加上去得到的结果可能有时候并不能帮助你的Agent学的更快，相反，有时候反而会把Agent带跑偏。这些其实都属于“眼镜蛇效应”：即之前某地政府为了让民众帮忙一起抓眼镜蛇，就给抓到蛇的那些人一定的奖赏，然后，民众们就开始养蛇了…… 这里注意在reward shaping过程中，你最终得到的是**你所鼓励或者抑制的行为**，而**不是你的意图**，除非你能保证这两者是match的。一般来讲，正向奖励一般的结果是使得AI更倾向于去**累积**这个奖励，除非你结束的那个奖励相当大否则他可能会故意不结束。而负向奖励一般的结果是使得AI更倾向于**尽快结束**来避免一直受到处罚。
>
> -- 知乎

 - **Policy invariance under reward transformations: Theory and application to reward shaping**

   奖励塑造是强化学习中常用的人为设计附加的奖励来指导智能体训练的方法，但是一些问题中人为设计的奖励函数常常导致智能体学习到非最优的策略。

   文章主要研究保证reward shaping最优策略不变的条件，结论是当附加奖励值可以表示为任意一个状态的势函数（Potential-based function，势函数被定义为状态到实数的映射 $  \phi: S \rightarrow  R $）的差分形式的时候，能保证最优策略不变。

   文章最后设计了基于距离和基于子目标的启发式奖励函数，并实验证明了其对缩减训练时间有很大作用。两个典型问题：
   （1）自动车从A到B的问题，当智能体向B走就给予正奖励，其余奖励0，可能会导致智能体学到在A附近“兜圈”的策略，原因是智能体远离B没有给予负奖励，当智能体“兜圈”时，凭借靠近B的部分就能持续获得奖励；
   （2）为了学会足球控球，当智能体碰到球就给予正奖励，可能导致智能体学习到在球附近“震荡”，快速靠近球然后远离再靠近，这样智能体也能持续不断获得奖励。

#### 1.3 机器人控制

- Motion Planning of Ladder Climbing for Humanoid Robots 

  <img src="C:\Users\孟一凌\AppData\Roaming\Typora\typora-user-images\image-20230419114956869.png" alt="image-20230419114956869" style="zoom: 50%;" />

  ​    综述中提到：Yoneda等人研制了一种专门为攀爬而设计的垂直爬梯类人机器人;它用钩状的手在攀爬过程中保持平衡，并使用力传感器来检测是否成功抓住了一个横档。与一般用途的类人机器人不同，这些特殊用途的攀爬机器人在操作、爬楼梯、从平地过渡到梯子以及从平地过渡到梯子等方面的能力有限或根本不存在。

  ​    提到了梯子的斜角最好为70°。目前设定的单阶梯子垂直高度为0.25，计算得到单阶梯子水平宽为0.09，暂取0.10。

  ​    上层设计手、脚在梯子上的离散点，下层为点到点的轨迹规划。需要满足实际机器人的各种约束（自碰撞、稳定性等等）。

  ​    上层的离散顺序：

  (1) placeHands: place two hands on a (chosen) rung. 

  (2)  placeLFoot: place left foot on the first rung. 

  (3) placeRFoot: place right foot on the first rung. 

  (4) moveLHand: lift left hand to the next higher rung. 

  (5) moveRHand: lift right hand to the next higher rung. 

  (6) moveLFoot: lift left foot to the next higher rung. 

  (7) moveRFoot: lift right foot to the next higher rung. 

  ​	
  
  

### 2. 研究意义

- 机器人的形态决定其功能。不同的形态又需要不同的控制器。在针对特定任务设计并优化机器人形态时，需要始终评估不同形态的性能。为众多形态分别设计控制器并进行实验是非常费时费力的。希望能够在仿真中构建控制器-形态联合优化的框架，减少优化时长，并提升最终效能。
- 在现有的关于机器人形态、结构优化研究中，研究对象大多是比较简单的机器人模型。本设计将模型更改为更复杂的Humanoid model。

## 二、 机器人控制器

### 1. 强化学习控制器

使用强化学习训练一个策略网络，作为机器人（agent）的控制器。可选择的网络有两个：SAC和PPO。

PPO的基本思想跟PG(Policy Gradient)算法一致，直接根据策略的收益好坏来调整策略。PPO是一种on-policy的算法，这类算法在训练时需要保证生成训练数据的policy与当前训练的policy一致，对于过往policy生成的数据难以再利用，所以在sample efficiency这条衡量强化学习算法的重要标准上难以取得优秀的表现。

在实际应用中，PPO算法总是可以获得一条稳定上升的episode-reward曲线，但曲线的斜率很低，如果要达到足够高的reward需要非常多的timestep（对于我们的任务可能是e7到e8的量级）。

SAC是一个off-policy，actor-critic算法。与其他RL算法最为不同的地方在于，SAC在优化策略以获取更高累计收益的同时，也会最大化策略的熵。SAC中的熵（entropy）可以理解为混乱度，无序度，随机程度，熵越高就表示越混乱，其包含的信息量就更多。假设有一枚硬币，不论怎么投都是正面朝上，则硬币的正反这一变量的熵就低，如果正反出现的概率都为0.5，则该变量的熵相对更高。将熵引入RL算法的好处为，可以让策略（policy）尽可能随机，agent可以更充分地探索状态空间S，避免策略早早地落入局部最优点（local optimum），并且可以探索到多个可行方案来完成指定任务，提高抗干扰能力。

#### 

###  2. Reward Shaping

强化学习中，一般鼓励使用宽泛的reward function。

若想让agent做出某些特定的动作，需要根据任务要求来精细地设计reward function。

**什么是Reward Shaping？**

> 当我们使用强化学习算法来解决一个特定的任务时，我们需要定义一个目标，也称为“奖励函数”或“回报函数”，来指导代理在环境中采取哪些行动。在某些情况下，这个奖励函数可能很难直接设计，或者可能存在一些不希望代理学到的副作用。这时候就需要使用奖励塑形（Reward shaping）技术来调整奖励函数，以更好地指导代理的学习。
>
> 奖励塑形的基本思想是通过修改奖励函数来改变代理在环境中的行为。**这种修改通常是基于对环境的先验知识或经验的利用，可以帮助代理更快地学习到好的策略，或避免不必要的代价。**例如，在一个围棋游戏中，如果我们只为代理在胜利时提供正奖励，那么代理可能会倾向于采取过于冒险的策略，以尽可能早地获得胜利。但如果我们为代理在每个步骤上的棋子数量提供适当的奖励，那么代理可能会更倾向于保持自己的棋子数量，并采取更稳健的策略。
>
> 总之，奖励塑形是一种非常有用的技术，可以帮助我们更好地指导代理在环境中学习到好的策略。

### 3.强化学习环境

#### 3.1 楼梯地形

对楼梯地形搭建强化学习环境，并打包为gym的env环境。

- 楼梯地形原版（3月）：

  Action space：

  为$19*1$的数组，表示全部电机的输出，限幅$[-0.4,0.4]$。

  Observation space：

  包括关节角度，关节速度，质心转动惯量等。**包括了agent在全局坐标系下的x，y坐标。**

  Reward：

  共分为三项：
  $$
  R = w_{forward}r_{foward} + w_{healthy}r_{healthy} + w_{stand}r_{stand} - w_{control}c_{control} - w_{contact}c_{contact}
  $$
  

  1. 前进奖励，包括了前进距离和前进速度：

  $$
  r_{forward} = w_{speed}v_x + w_{distace}x
  $$

  2. 存活奖励，当机器人被判断为存活时，始终能够得到该奖励。

  3. 站立奖励。计算机器人躯干坐标系的z轴与世界坐标系的z轴的重合程度，即姿态直立程度。映射到$[0,1]$
     $$
     r_{stand} = (\vec{z_{world}}\cdot\vec{z_{agent}} + 1) /2
     $$

  同时包括两项cost，接触cost和控制cost。

- **楼梯地形新版：**

  Action space：

  未改动，仍为$19*1$的数组，表示全部电机的输出，限幅$[-0.4,0.4]$。

  Observation space：

  包括关节角度，关节速度，质心转动惯量等。**移除了agent在全局坐标系下的x，y坐标**。**添加了一个$7*1$的楼梯相对位置数组。**具体表示为：以agent当前的全局$x,y,z$坐标为中心，x正方向每隔0.1取一个点，共取5个；x负方向隔0.1取1个点；加上agent自身的中心点，形成一列七个$y，z$坐标相同的点。读取每个点距离楼梯地面的距离。

  Reward：

  共分为三项：
  $$
  R = w_{forward}r_{foward} + w_{healthy}r_{healthy} + w_{stand}r_{stand} - w_{control}c_{control} - w_{contact}c_{contact}
  $$


  1. 前进奖励，**包括前进速度，移除了前进距离**：

$$
  r_{forward} = v_x
$$

  2. 存活奖励，当机器人被判断为存活时，始终获得该奖励，**但更改为了值很小的负数**。

  3. 站立奖励。计算机器人躯干坐标系的z轴与世界坐标系的z轴的重合程度，即姿态直立程度。**增加了方向项，即机器人躯干坐标系x轴与世界坐标系的x轴的重合程度。**
     $$
     r_{stand} = (\vec{z_{world}}\cdot\vec{z_{agent}} + 1 + \vec{x_{world}}\cdot\vec{x_{agent}} + 1) /4
     $$

  同时包括两项cost，接触cost和控制cost。

关于4月新版环境的一些改动的说明：

- 在观测空间里移除了全局x，y坐标；同时在奖励函数中移除了x方向位移带来的reward。

  > 因为在实际行走中，机器人的运动应该为稳定持久 ，不随行走距离增大而发生改变的，因此在观测中，不应包括机器人在仿真世界中的前进距离 ，否则待训练智能体的决策将受到此信息的干扰，当机器人其他信息相同，只有前进距离不同时 ，会出现不同的动作 ，影响持久性。  
  >
  > --强化学习在双足机器人步态控制中的应用  ，张惟宜  

  ​     从本质上来说，楼梯地形和平坦地面没有本质区别，都是在无限延展的地形中稳定的走下去。楼梯地形无非是增加了一些台阶，且这些台阶沿x方向（前进方向）是周期性分布的。因此，在观测空间中添加全局x坐标项，会导致训练变得更复杂：对于agent来说，由于全局x坐标观测量的存在，走上第7个台阶和走上第1个台阶变得不一样了，需要重新学习。而实际上并没有什么不一样。

  ​     同样的，在奖励函数中添加关于x的位移项，也可能会引发类似的问题：由于在楼梯高处的奖励函数变得更高，agent不会使用在低处楼梯相同的策略来走上高处楼梯，可能会使用一些简单的策略来获得更高的位移奖励，比如一个鱼跃然后摔倒。这会不可预测地影响训练。

  ​    走楼梯和平地行走都是在学习一个周期性的、能够保持平衡的步态。原先的观测空间和奖励函数的设定将agent要学习的任务隐式地更改为“尽可能地快速移动到更高的位置”。这也是为什么先前训练得到的结果在楼梯高处表现出的平衡性远远低于在楼梯低处。

- 为什么要将生存奖励改为负数？

  由于对模型添加了带有踝关节的脚部，现在的机器人拥有了站立功能。在训练中，若生存奖励为正数，很容易得到一个站立不动且获得大量生存奖励的机器人。尽管设定了速度过小的终止判定，仍无法从根本上解决这一问题。于是将生存奖励调整成为了较小的负数。

- 为什么要在观测空间中添加楼梯相对位置数组？

  楼梯地形的目标任务虽然在本质上和平面地形 是一样的，但是仍然有周期性的地形变化。agent需要获得周期性的地形信息。



#### 3.2 梯子地形

- **原版（3月）梯子地形：**

  梯子地形的reward function目前基本的形式与楼梯地形相同：

  其中包括：

  1. 前进项：沿x位置，沿x速度，沿z位置，沿z速度
  2. 接触项：当agent的手或者脚与梯子接触时，给予高额reward。每个手/脚接触每层梯子的reward仅给一次，防止agent将手放在梯子上不动。越高处的梯子给的reward越高。
  3. 姿态项：躯干站立越直，奖励越高。
  4. 生存惩罚：当agent存活时，给予负的生存奖励，逼迫其进行探索。

  $$
  R = w_{forward}r_{foward} + w_{healthy}r_{healthy} + w_{stand}r_{stand} - w_{control}c_{control} - w_{contact}c_{contact}
  $$

  其中的前进奖励项更改为向上高度的奖励：
  $$
  r_{forward} = w_{speed}v_x + w_{height}z
  $$
  作为调整，调低了前进速度权重$w_{speed}$，调高了高度奖励$w_{height}$，调低了生存奖励$w_{healthy}$，调低了直立奖励$w_{stand}$。目的是进一步鼓励机器人沿着梯子上升。

  问题也非常明显。相比于走楼梯，爬梯子是一系列更为复杂的动作，需要全身协调配合。因此，尽管有着向上高度的奖励函数来引导它向上，但是agent并不知道如何向上攀爬。

  ```python
      @property
      def contact_reward(self):
          '''
          - 读取contact信息。
          - 扫描contact数组，寻找其中是否有‘手-梯子’，‘脚-梯子’的碰撞对。注意geom1既可能是梯子		也可能是手。
          - 如果有，将这一对碰撞对保存下来。若该碰撞对已存在，则跳过，不获得奖励函数。
          - 根据梯子的阶数，赋予奖励值。梯子越高，奖励值越高
          '''
          # 计算接触reward
          reward = 0
          if self.terrain_type == 'ladders':
              contact = list(self.sim.data.contact)  # 读取一个元素为mjContact的结构体				数组
              ncon = self.sim.data.ncon # 碰撞对的个数
              for i in range(ncon): # 遍历所有碰撞对
                  con = contact[i]
                  # 判断ladder是否参与碰撞
                  if 'ladder' in self.geomdict[con.geom1]+self.geomdict[con.geom2]:
                      ladder = self.geomdict[con.geom1] if 'ladder' in self.geomdict[con.geom1] else self.geomdict[con.geom2]
                      # 判断是手还是脚
                      if 'hand' in self.geomdict[con.geom1]+self.geomdict[con.geom2]:
                          # 区分左右手加分
                          limb = 'right_hand' if 'right' in self.geomdict[con.geom1]+self.geomdict[con.geom2] else 'left_hand'
                      elif 'foot' in self.geomdict[con.geom1]+self.geomdict[con.geom2]:
                          limb = 'right_foot' if 'right' in self.geomdict[con.geom1]+self.geomdict[con.geom2] else 'left_foot'
                      else: # 若非手脚，跳过
                          continue
                  else:
                      continue
                  cont_pair = (limb,ladder)
                  if cont_pair in self.already_touched: # 判断是否曾经碰撞过
                      continue
                  else:
                      ladder_num = int(ladder[6:])
                      # 手部仅可碰撞到6阶以上时有奖励分
                      if 'hand' in limb and ladder_num < 5:
                          continue
                      reward = reward + 50*ladder_num
                      self.already_touched.append(cont_pair)
  
          if self.terrain_type == 'steps':
              reward = 0
          contact_reward = reward * self._contact_reward_weight           
          return contact_reward
  
  ```

  <img src="C:\Users\孟一凌\AppData\Roaming\Typora\typora-user-images\image-20230404152020410.png" alt="image-20230404152020410" style="zoom: 33%;" />

- **新版（4月）梯子地形奖励函数：**

  Observation Space：

  添加对梯子的相对位置信息读取：

  确定agent当前触碰到的最低一级梯子（地板和第一级梯子优先级相同），最低一级梯子往上数三个梯子，取agent到这三个梯子中心的x方向距离，z方向距离，总共6个数据。

  Action Space：

  无改动，仍然是 $19\times1$的动作空间。

  Reward：

  共四项：
  $$
  R = w_{forward}r_{foward} + w_{healthy}r_{healthy} + w_{stand}r_{stand} + w_{contact}r_{contact} - w_{control}c_{control} - w_{contact}c_{contact}
  $$

  1. 前进项：只保留x方向速度和z方向速度，其中z方向速度占比更大。
     $$
     r_{forward}= v_x + 2v_z
     $$

  2. 生存项：当agent存活时，给予负的生存奖励，逼迫其进行探索。为-0.2。当触碰到最高处的阶梯时，终止。

  3. 接触项：agent手/脚接触到梯子时

     ​    吸收楼梯环境的经验，消去和全局位置相关的特征，以提升训练稳定性。对于不同的台阶，均给出相同的reward值。这个值不应太大，在先前的试验中，若离散reward给的太高，会严重影响agent的exploration效率，过早收敛。

     ​    根据李宏毅Reward shaping的主要思想，$R(s_t)$函数应具有单一方向的持续上升的梯度，此方向为先验知识指导agent学习的方向。既然难以指导agent学会爬梯子的动作，不如惩罚agent不爬梯子/向下走。

     由此，最终接触项设计如下：

     - 单次接触reward设计为10。
     - 若某个肢体（左右手、左右脚）碰到了比先前碰到过的最高阶梯要低的阶梯，给予-200的reward。
     - 重复接触不计算reward。

  4. 姿态项：移除了直立奖励，爬梯子的过程中应该很难保持直立。添加了朝向奖励。


---

训练记录：

1. 新版reward训练，失败

2. 发现姿态项给的reward值太大，失败

3. 移除了姿态项reward，依旧失败

4. 将梯子的角度放缓，添加摔倒检测，失败

5. 移除摔倒检测，失败

6. 添加了站立监测，将生存奖励改为0，失败，甚至原地起跳。

7. 发现一个bug，将肢体从高处放到低处会疯狂扣分而不是只扣一次，修正。将速度项的x权重加大，z权重减少，因为刚开始爬梯子时有弯腰动作。失败。学会了蛤蟆跳等死。

8. 继续上一条改动，将梯子斜度提升（x间距0.16），仍然失败。蛤蟆跳。

   <img src="C:\Users\孟一凌\AppData\Roaming\Typora\typora-user-images\image-20230420135607018.png" alt="image-20230420135607018" style="zoom: 20%;" /><img src="C:\Users\孟一凌\AppData\Roaming\Typora\typora-user-images\image-20230420135719561.png" alt="image-20230420135719561" style="zoom:40%;" />

9. 4月20日，准备对手部添加关节。意味着先前全部的模型将被弃用（虽然本来也没有能用的）。





#### 3.3 离散Reward Function在本任务中的具体实现

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

2. One Sci/Ei paper published with the results

