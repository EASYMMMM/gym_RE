'''
humanoidCustom.py
复制自 anaconda3\envs\GYM\Lib\site-packages\gym\envs\mujoco\humanoid_v3.py
在此基础上自定义mujoco ENV环境
'''

import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
import os
from scipy.spatial.transform import Rotation as R
import sys,os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from generateXML import HumanoidXML


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 2.0)),
    "elevation": -20.0,
}

HORIZONTAL_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 10.0,
    "lookat": np.array((0.0, 0.0, 1.0)),
    "elevation": 0.0,
}

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:3].copy()


class HumanoidCustomEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        xml_file="humanoid_custom.xml",
        terrain_type="steps",
        forward_speed_reward_weight=0.8,
        forward_distance_reward_weight=1.5,
        ctrl_cost_weight=0.1,
        contact_cost_weight=5e-7,
        contact_cost_range=(-np.inf, 10.0),
        healthy_reward= -0.2,                     # 存活奖励
        stand_reward_weight = 1.0,              # 站立奖励
        contact_reward_weight = 1.0,            # 梯子/阶梯 接触奖励
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 5.0),
        reset_noise_scale=1e-2,
        camera_config = "horizontal",
        exclude_current_positions_from_observation=False,  # Flase: 使obs空间包括躯干的x，y坐标; True: 不包括
    ):
        utils.EzPickle.__init__(**locals())

        # set the terrain, generate XML file
        self.terrain_type = terrain_type        
        terrain_list = ('default','steps','ladders') # 默认平地，台阶，梯子
        assert self.terrain_type in terrain_list, 'ERROR:Undefined terrain type'  
        xml_name = 'humanoid_exp_v1.xml'
        self.xml_model = HumanoidXML(terrain_type=self.terrain_type)
        self.xml_model.write_xml(file_path=f"gym_custom_env/assets/{xml_name}")
        dir_path = os.path.dirname(__file__)
        xml_file_path = f"{dir_path}\\assets\\{xml_name}"
        
        self._forward_speed_reward_weight = forward_speed_reward_weight
        self._forward_distance_reward_weight = forward_distance_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._contact_cost_range = contact_cost_range
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._stand_reward_weight = stand_reward_weight
        self._contact_reward_weight = contact_reward_weight
        self._reset_noise_scale = reset_noise_scale
        
        self.camera_config = {
            "defalt":DEFAULT_CAMERA_CONFIG,
            "horizontal":HORIZONTAL_CAMERA_CONFIG,
        }[camera_config]

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self.x_velocity = 0                         # 质心沿x速度
        self._walking_counter = 0                   # 判定正常前进计数器
        self.already_touched =[]                    # 记录已经碰撞过的geom对
        ladders = ['ladder' + str(i) for i in range(1, 12)]
        contact_pairs = [('right_hand', ladder) for ladder in ladders] + [('left_hand', ladder) for ladder in ladders] + \
                        [(ladder, 'right_hand') for ladder in ladders] + [(ladder, 'left_hand') for ladder in ladders] + \
                        [('right_foot_geom', ladder) for ladder in ladders] + [('left_foot_geom', ladder) for ladder in ladders] + \
                        [(ladder, 'right_foot_geom') for ladder in ladders] + [(ladder, 'left_foot_geom') for ladder in ladders]
        self.contact_list = contact_pairs

        print("============ HUMANOID CUSTOM ENV ============")
        print(f"=====terrain type:{self.terrain_type}=====")
        mujoco_env.MujocoEnv.__init__(self, xml_file_path, 5)
        self.geomdict = self.get_geom_idname()      # geom id name字典

    @property
    def is_walking(self):
        # 判断机器人是否正常行走，若连续10步速度小于某值，停止
        _is_walking = True

        # 对于阶梯地形，未进入阶梯时，直接返回True
        if self.terrain_type == 'steps':
            if self.sim.data.qpos[0] < 0:   
                return _is_walking

        if self.x_velocity < 0.1: 
            if self._walking_counter > 100:
                _is_walking = False
                self._walking_counter = 0
            else:
                self._walking_counter = self._walking_counter + 1
        return _is_walking
        

    @property
    def healthy_reward(self):
        # 机器人正常运行的reward值，_healthy_reward默认为5，即正常训练时healthy_reward = 5
        # 当机器人前进速度小于0.05时，判定为摔倒，停止训练。
        return (
            ( float((self.is_healthy or self._terminate_when_unhealthy) and  self._is_walking ) )
            * self._healthy_reward
        )
    
    @property
    def forward_reward(self):
        # 计算前进奖励 

        # 楼梯地形
        # 前进奖励 = 速度权重*前进速度 + 距离权重*前进距离
        # 计算前进距离时 +1，因为机器人起始点在x=-1
        if self.terrain_type == 'steps':
            forward_reward = self._forward_speed_reward_weight * self.x_velocity + self._forward_distance_reward_weight * (self.sim.data.qpos[0] + 1) # self.sim.data.qpos[0]: x coordinate of torso (centre)
        
        # 梯子地形
        # 前进奖励 = 速度权重*前进速度 + 5*距离权重*高度
        if self.terrain_type == 'ladders':
            forward_reward = self._forward_speed_reward_weight * self.x_velocity + 3*self._forward_distance_reward_weight * (self.sim.data.qpos[2]-1.4) # self.sim.data.qpos[0]: x coordinate of torso (centre)
    
        return forward_reward

    @property
    def stand_reward(self):
        # 计算直立奖励
        quatanion = self.sim.data.qpos[3:7]
        Rm = R.from_quat(quatanion)  # Rotation matrix
        rotation_matrix = Rm.as_matrix()
        # 提取躯干的z轴方向向量
        vertical_direction = np.array([0, 0, 1])
        body_z_axis = rotation_matrix.dot(vertical_direction)
        # 计算z轴方向向量与竖直向上方向向量的点积
        dot_product = np.dot(body_z_axis, vertical_direction)
        # 将点积映射到[0, 1]范围内的奖励值
        # TODO: 这里计算出的点积取负。实验证明站立时点积为-1，暂时还不知道是为什么
        stand_reward = self._stand_reward_weight * (( - dot_product + 1.0) / 2.0)  
        
        # 如果是楼梯地形，直接返回reward值。如果是阶梯地形，乘以系数0.5
        if self.terrain_type == "ladders":
            stand_reward = stand_reward * 0.1
        return stand_reward

    def control_cost(self, action):
        # 控制花费。所有控制量的开方和。
        control_cost = self._ctrl_cost_weight * np.sum(np.square(self.sim.data.ctrl))
        return control_cost

    @property
    def contact_cost(self):
        contact_forces = self.sim.data.cfrc_ext
        contact_cost = self._contact_cost_weight * np.sum(np.square(contact_forces))
        min_cost, max_cost = self._contact_cost_range
        contact_cost = np.clip(contact_cost, min_cost, max_cost)
        return contact_cost

    @property
    def is_healthy(self):
        # 机器人状态是否正常，通过qpos[2]的z轴位置判断（是否跌倒）
        min_z, max_z = self._healthy_z_range
        is_standing = min_z < self.sim.data.qpos[2] < 10  #  self.sim.data.qpos[2]: z-coordinate of the torso (centre)
        is_inthemap = self.sim.data.qpos[0] < 4.7         #  机器人仍然在阶梯范围内   
        is_healthy  = is_standing and is_inthemap
        return is_healthy


    @property
    def done(self):
        # episode是否结束的标志，在step()函数中返回
        # 如果机器人的状态是unhealthy或摔倒，则done = True，训练终止
        done = ( (not self.is_healthy) or (not self._is_walking) ) if self._terminate_when_unhealthy else False
        return done

    @property
    def contact_reward(self):
        '''
        TODO:
        - 读取contact信息。
        - 扫描contact数组，寻找其中是否有‘手-梯子’，‘脚-梯子’的碰撞对。注意geom1既可能是梯子也可能是手。
        - 如果有，将这一对碰撞对保存下来。若该碰撞对已存在，则跳过，不获得奖励函数。
        - 根据梯子的阶数，赋予奖励值。梯子越高，奖励值越高
        '''
        # 计算接触reward
        reward = 0
        if self.terrain_type == 'ladders':
            contact = list(self.sim.data.contact)  # 读取一个元素为mjContact的结构体数组
            ncon = self.sim.data.ncon # 碰撞对的个数
            for i in range(ncon): # 遍历所有碰撞对
                con = contact[i]
                # 判断ladder是否参与碰撞
                if 'ladders' in self.geomdict[con.geom1]+self.geomdict[con.geom2]:
                    ladder = self.geomdict[con.geom1] if 'ladders' in self.geomdict[con.geom1] else self.geomdict[con.geom2]
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
                    if 'hand' in limb and ladder_num < 6:
                        continue
                    reward = reward + 25*ladder_num
                    self.already_touched.append(cont_pair)
                    
        if self.terrain_type == 'steps':
            reward = 0
        contact_reward = reward * self._contact_reward_weight           
        return contact_reward

    def _get_obs(self):
        # obs空间
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        com_inertia = self.sim.data.cinert.flat.copy()
        com_velocity = self.sim.data.cvel.flat.copy()

        actuator_forces = self.sim.data.qfrc_actuator.flat.copy()
        external_contact_forces = self.sim.data.cfrc_ext.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        return np.concatenate(
            (
                position,
                velocity,
                com_inertia,
                com_velocity,
                actuator_forces,
                #external_contact_forces,
            )
        )

    def get_geom_idname(self,):
        geomdict = {}
        for i in range(self.model.ngeom):
            geomdict[i] = self.model.geom_id2name(i)
        return geomdict


    def print_obs(self):
        # obs空间
        position = self.sim.data.qpos.flat.copy()
        print('position shape:')
        print(position.shape)

        velocity = self.sim.data.qvel.flat.copy()
        print('velocity shape:')
        print(velocity.shape)

        com_inertia = self.sim.data.cinert.flat.copy()
        print('com_inertia shape:')
        print(com_inertia.shape)        

        com_velocity = self.sim.data.cvel.flat.copy()
        print('com_velocity shape:')
        print(com_velocity.shape)  

        actuator_forces = self.sim.data.qfrc_actuator.flat.copy()
        print('actuator_forces shape:')
        print(actuator_forces.shape)  
        external_contact_forces = self.sim.data.cfrc_ext.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]


    def step(self, action):
        '''
        step(self, action) -> observation, reward, done, info
        在父类mujoco_env初始化时，会调用该函数，并根据返回的observation来确定observation space。
        因此更改返回值中的observation，同时可更改该env的observation space。
        '''
        xyz_position_before = mass_center(self.model, self.sim)
        self.do_simulation(action, self.frame_skip)
        xyz_position_after = mass_center(self.model, self.sim)

        # dt为父类mujoco_env中的一个@property函数 
        xy_velocity = (xyz_position_after[0:2] - xyz_position_before[0:2]) / self.dt
        x_velocity, y_velocity = xy_velocity
        self.x_velocity = x_velocity

        # 是否仍在前进，只调用一次，防止反复调用is_walking出错
        self._is_walking = self.is_walking

        # cost值 控制cost + 接触力cost
        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        # reward值 
        # _forward_reward_weight 默认为1.25, 将x轴位移添加到奖励值中，鼓励前进。
        forward_reward = self.forward_reward
        healthy_reward = self.healthy_reward
        stand_reward   = self.stand_reward
        contact_reward = self.contact_reward

        rewards = forward_reward + healthy_reward + stand_reward + contact_reward
        costs = ctrl_cost + contact_cost

        observation = self._get_obs()
        reward = rewards - costs
        done = self.done
        info = {
            "reward_linvel": forward_reward,
            "reward_quadctrl": -ctrl_cost,
            "reward_alive": healthy_reward,
            "reward_impact": -contact_cost,
            "x_position": xyz_position_after[0],
            "y_position": xyz_position_after[1],
            "z_position": xyz_position_after[2],
            "xyz_position": xyz_position_after,
            "distance_from_origin": np.linalg.norm(xyz_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
            "is_healthy": self.is_healthy,
            "is_walking": self._is_walking,
        }

        return observation, reward, done, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in self.camera_config.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
