# Humanoid Robot Evolution

##  Reward

强化学习中，一般鼓励使用宽泛的reward function。

若想让agent做出某些特定的动作，需要利用expert knowledge来精细地设计reward function。

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
