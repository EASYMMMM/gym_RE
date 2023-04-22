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

def set_ladder_height():
    return {'flatfloor':0,
            'ladder1':1,
            'ladder2':2,
            'ladder3':3,
            'ladder4':4,
            'ladder5':5,
            'ladder6':6,
            'ladder7':7,
            'ladder8':8,
            'ladder9':9,
            'ladder10':10,  
            'ladder11':11,                         
            }

class HumanoidCustomEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        xml_file="humanoid_custom.xml",
        terrain_type="steps",
        forward_speed_reward_weight=1.5,
        forward_distance_reward_weight=1.5,
        ctrl_cost_weight=0.1,
        contact_cost_weight=5e-7,
        contact_cost_range=(-np.inf, 10.0),
        healthy_reward= -0.2,                     # 存活奖励
        posture_reward_weight = 1,              # 站立奖励
        contact_reward_weight = 1.0,            # 梯子/阶梯 接触奖励
        terminate_when_unhealthy=True,
        healthy_z_range=(0.9, 5.0),
        reset_noise_scale=1e-2,
        camera_config = "horizontal",
        single_contact_reward = 10,
        exclude_current_positions_from_observation=False,  # Flase: 使obs空间包括躯干的x，y坐标; True: 不包括
    ):
        utils.EzPickle.__init__(**locals())

        # set the terrain, generate XML file
        self.terrain_type = terrain_type        
        terrain_list = ('default','steps','ladders') # 默认平地，台阶，梯子
        assert self.terrain_type in terrain_list, 'ERROR:Undefined terrain type'  
        xml_name = 'humanoid_exp.xml'
        self.xml_model = HumanoidXML(terrain_type=self.terrain_type)
        self.xml_model.write_xml(file_path=f"gym_custom_env/assets/{xml_name}")
        dir_path = os.path.dirname(__file__)
        xml_file_path = f"{dir_path}\\assets\\{xml_name}"
        
        self._forward_speed_reward_weight = forward_speed_reward_weight
        self._forward_distance_reward_weight = forward_distance_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight if terrain_type in 'default'+'steps' else 0.2*ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._contact_cost_range = contact_cost_range
        self._healthy_reward = -0.1 if terrain_type in 'default'+'steps' else 0
        self._single_contact_reward = single_contact_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._posture_reward_weight = posture_reward_weight
        self._contact_reward_weight = contact_reward_weight
        self._reset_noise_scale = reset_noise_scale
        self.camera_config = {
            "defalt":DEFAULT_CAMERA_CONFIG,
            "horizontal":HORIZONTAL_CAMERA_CONFIG,
        }[camera_config]

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self.__init_counter()

        print("============ HUMANOID CUSTOM ENV ============")
        print(f"=====terrain type:{self.terrain_type}=====")
        mujoco_env.MujocoEnv.__init__(self, xml_file_path, 5)
        self.geomdict = self.get_geom_idname()      # geom id name字典

    def __init_counter(self):
        # 初始化/清零需要用到的储存数组
        self.x_velocity = 0                         # 质心沿x速度
        self.z_velocity = 0                         # 质心沿z速度
        self._walking_counter = 0                   # 判定正常前进计数器
        self.already_touched =[]                    # ladder:记录已经碰撞过的geom对
        self.limb_position = {'right_hand':0,    # ladder:记录手脚到达过的最高位置
                              'left_hand':0,
                              'right_foot':0,
                              'left_foot':0}
        self.ladder_height = set_ladder_height()
        self.ladder_up = True                       # 是否在梯子上保持上升
        self.contact_reward_sum = 0
        self.healthy_reward_sum = 0
        self.forward_reward_sum = 0
        self.posture_reward_sum = 0
        self.contact_cost_sum = 0
        self.control_cost_sum = 0
        self.ladder_task = 0        # 爬梯子任务分解，当前任务序号 
        self.ladder_task_flag = { 0:{'right_hand':False, 'left_hand':False} # 分解任务中用到的标志
                                    }  
        
    @property
    def is_walking(self):
        # 判断机器人是否正常行走，若连续10步速度小于某值，停止
        _is_walking = True
        if self.terrain_type == 'ladders':
            threshold = 200
        else:
            threshold = 100
        # 对于阶梯地形，未进入阶梯时，直接返回True
        #if self.terrain_type == 'steps':
        #    if self.sim.data.qpos[0] < 0:   
        #        return _is_walking

        if self.x_velocity < 0.1: 
            if self._walking_counter > threshold:
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
        # 前进奖励 = 速度权重*前进速度
        if self.terrain_type == 'steps':
            forward_reward = self._forward_speed_reward_weight * self.x_velocity  # self.sim.data.qpos[0]: x coordinate of torso (centre)
       
        # 梯子地形
        # 前进奖励 = 速度权重*前进速度 
        if self.terrain_type == 'ladders':
            forward_reward = self._forward_speed_reward_weight * (2*self.x_velocity + self.z_velocity) 
        return forward_reward

    @property
    def posture_reward(self):
        # 姿态奖励 = 直立 + 方向朝前

        quatanion = self.sim.data.qpos[3:7]
        Rm = R.from_quat(quatanion)  # Rotation matrix
        rotation_matrix = Rm.as_matrix()
        # 提取躯干的z轴方向向量
        vertical_direction = np.array([0, 0, 1])
        body_z_axis = rotation_matrix.dot(vertical_direction)
        # 计算z轴方向向量与竖直向上方向向量的点积
        z_dot_product = np.dot(body_z_axis, vertical_direction)
        # 提取躯干的x轴方向向量
        forward_direction = np.array([1, 0, 0])
        body_x_axis = rotation_matrix.dot(forward_direction)
        # 计算x轴方向向量与竖直向上方向向量的点积
        x_dot_product = np.dot(body_x_axis, forward_direction)
        # 将点积映射到[0, 1]范围内的奖励值
        # 楼梯地形同时考虑姿态和朝向
        if self.terrain_type == "steps" or self.terrain_type == "default":
            # TODO: 这里计算出的点积取负。实验证明站立时点积为-1，暂时还不知道是为什么
            reward = self._posture_reward_weight * ( (( - z_dot_product + 1.0) / 2.0) + (( x_dot_product + 1.0) / 2.0) )/2
        # 阶梯地形只考虑朝向。为了防止持续获得奖励，朝向偏差太多时扣分，正向不额外给分。
        if self.terrain_type == "ladders":
            reward = 0 if self._posture_reward_weight*(( x_dot_product + 1.0) / 2.0) > 0.8 else -1
        return reward

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
    def _not_fallen(self):
        # 检测是否摔倒需要提前终止
        # 返回False：需要提前终止

        contact = list(self.sim.data.contact)  # 读取一个元素为mjContact的结构体数组
        ncon = self.sim.data.ncon # 碰撞对的个数
        not_fallen = True
        if self.terrain_type == 'ladders':
            # 梯子地形中，躯干碰到梯子即为摔倒
            for i in range(ncon): # 遍历所有碰撞对
                con = contact[i]
                # 判断ladder是否参与碰撞
                if 'ladder' in self.geomdict[con.geom1]+self.geomdict[con.geom2]:
                    # if 'torso' in self.geomdict[con.geom1]+self.geomdict[con.geom2]: # 躯干
                    #    not_fallen = False
                    if 'waist' in self.geomdict[con.geom1]+self.geomdict[con.geom2]: # 腰
                        not_fallen = False
                    if 'head' in self.geomdict[con.geom1]+self.geomdict[con.geom2]: # 头
                        not_fallen = False
                    if 'pelvis' in self.geomdict[con.geom1]+self.geomdict[con.geom2]: # 腰
                        not_fallen = False
            result = self.ladder_up and not_fallen
            return result
        else:
            return True
        # 检测是否摔倒
        contact = list(self.sim.data.contact)  # 读取一个元素为mjContact的结构体数组
        ncon = self.sim.data.ncon # 碰撞对的个数
        fallen = True
        if self.terrain_type == 'steps':
            # 台阶地形中，除了脚步以外的肢体碰撞到台阶，即为摔倒
            for i in range(ncon): # 遍历所有碰撞对
                con = contact[i]
                # 判断ladder是否参与碰撞
                if 'step' in self.geomdict[con.geom1]+self.geomdict[con.geom2]:
                    if 'foot' in self.geomdict[con.geom1]+self.geomdict[con.geom2]:
                         continue
                    else:
                        fallen = False
        return fallen
                
    @property
    def is_healthy(self):
        # 机器人状态是否正常，通过qpos[2]的z轴位置判断（是否跌倒）
        min_z, max_z = self._healthy_z_range
        z = self.sim.data.qpos[2]
        if self.terrain_type == 'ladders':
            min_z = 0.9
            lowest_ladder = self.limb_position['left_foot'] if self.limb_position['left_foot'] < self.limb_position['right_foot'] else self.limb_position['right_foot']
            lowest_ladder_height = lowest_ladder * self.xml_model.ladder_positions[0][2]
            z = self.sim.data.qpos[2] - lowest_ladder_height
            if self.limb_position['right_hand'] == 11 or self.limb_position['left_hand'] == 11:
                is_inthemap = False
            else:
                is_inthemap = True    
        if self.terrain_type == 'steps':
            step_pos = self._get_steps_pos()
            z = step_pos[1]
            is_inthemap = self.sim.data.qpos[0] < 6.6         #  机器人仍然在阶梯范围内  
        is_standing = min_z < z < 10  #  self.sim.data.qpos[2]: z-coordinate of the torso (centre)
        is_healthy  = is_standing and is_inthemap and self._not_fallen
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
        - 读取contact信息。
        - 扫描contact数组，寻找其中是否有‘手-梯子’，‘脚-梯子’的碰撞对。注意geom1既可能是梯子也可能是手。
        - 如果有，将这一对碰撞对保存下来。若该碰撞对已存在，则跳过，不获得奖励函数。
        - 根据梯子的阶数，赋予奖励值。梯子越高，奖励值越高
        '''
        # 计算接触reward
        reward = 0
        limb_sensor_state = {'right_hand':100,    # 当前的肢体末端的有效接触情况，100表示悬空
                    'left_hand' :100,
                    'right_foot':100,
                    'left_foot' :100}
        if self.terrain_type == 'ladders':
            # FIXME : 调试梯子任务分解时，临时更改 
            return self.ladder_task_reward
            contact = list(self.sim.data.contact)  # 读取一个元素为mjContact的结构体数组
            ncon = self.sim.data.ncon # 碰撞对的个数
            for i in range(ncon): # 遍历所有碰撞对
                con = contact[i]
                # 判断ladder/floor是否参与碰撞
                if ('ladder' in self.geomdict[con.geom1]+self.geomdict[con.geom2]) or ( 'floor' in self.geomdict[con.geom1]+self.geomdict[con.geom2] ):
                    ladder = self.geomdict[con.geom1] if 'ladder' in self.geomdict[con.geom1] else self.geomdict[con.geom2]
                    ladder = self.geomdict[con.geom1] if 'floor' in self.geomdict[con.geom1] else self.geomdict[con.geom2]
                    # 判断是手还是脚
                    if 'hand' in self.geomdict[con.geom1]+self.geomdict[con.geom2]:
                        # 区分左右手加分
                        limb = 'right_hand' if 'right' in self.geomdict[con.geom1]+self.geomdict[con.geom2] else 'left_hand'
                    elif 'foot' in self.geomdict[con.geom1]+self.geomdict[con.geom2]:
                        limb = 'right_foot' if 'right' in self.geomdict[con.geom1]+self.geomdict[con.geom2] else 'left_foot'
                    else: # 若非手脚，跳过
                        continue
                if 'sensor' in self.geomdict[con.geom1]+self.geomdict[con.geom2]:
                    # 更新当前的有效接触情况
                    limb_sensor_state[limb] = self.ladder_height[ladder]                    
                else:
                    continue
                
                cont_pair = (limb,ladder) 
                # 若当前碰到的阶梯高度比先前碰到的要低
                if self.ladder_height[ladder] < self.limb_position[limb]:
                    reward += -50
                    self.ladder_up = False
                    self.limb_position[limb] = self.ladder_height[ladder] # 防止反复扣分
                else:
                    self.limb_position[limb] = self.ladder_height[ladder]

                if cont_pair in self.already_touched: # 判断是否曾经碰撞过
                    continue
                else: # 初次碰撞，计算reward
                    if ladder == 'flatfloor': continue
                    ladder_num = int(ladder[6:])
                    # 手部仅可碰撞到6阶以上时有奖励分
                    if 'hand' in limb and ladder_num < 5:
                        continue
                    reward = reward + self._single_contact_reward
                    self.already_touched.append(cont_pair)

        if self.terrain_type == 'steps':
            reward = 0
        contact_reward = reward * self._contact_reward_weight    

        return contact_reward

    @property
    def ladder_task_reward(self):
        '''
        TODO 梯子任务分解
        根据规划的爬梯子离散动作进行奖励函数设计
        (0) placeHands: place two hands on a (chosen) rung. 
        (1) placeLFoot: place left foot on the first rung. 
        (2) placeRFoot: place right foot on the first rung. 
        (3) moveLHand: lift left hand to the next higher rung. 
        (4) moveRHand: lift right hand to the next higher rung. 
        (6) moveLFoot: lift left foot to the next higher rung. 
        (7) moveRFoot: lift right foot to the next higher rung. 
        将有效碰撞体积限制为手掌、脚掌中心。
        ''' 
        self._forward_speed_reward_weight = 0.5
        self._healthy_reward = 0.2
        contact = list(self.sim.data.contact)  # 读取一个元素为mjContact的结构体数组
        ncon = self.sim.data.ncon # 碰撞对的个数
        reward = 0
        limb_sensor_state = {'right_hand':100,    # 当前的肢体末端的有效接触情况，100表示悬空
                            'left_hand' :100,
                            'right_foot':100,
                            'left_foot' :100}

        # 更新当前手，脚所处阶梯数
        for i in range(ncon): # 遍历所有碰撞对
            con = contact[i]
            # 判断ladder/floor是否参与碰撞
            if ('ladder' in self.geomdict[con.geom1]+self.geomdict[con.geom2]) or ( 'floor' in self.geomdict[con.geom1]+self.geomdict[con.geom2] ):
                ladder = self.geomdict[con.geom1] if 'ladder' in self.geomdict[con.geom1] else self.geomdict[con.geom2]
                ladder = self.geomdict[con.geom1] if 'floor' in self.geomdict[con.geom1] else self.geomdict[con.geom2]
                # 判断是手还是脚
                if 'hand' in self.geomdict[con.geom1]+self.geomdict[con.geom2]:
                    # 区分左右手加分
                    limb = 'right_hand' if 'right' in self.geomdict[con.geom1]+self.geomdict[con.geom2] else 'left_hand'
                elif 'foot' in self.geomdict[con.geom1]+self.geomdict[con.geom2]:
                    limb = 'right_foot' if 'right' in self.geomdict[con.geom1]+self.geomdict[con.geom2] else 'left_foot'
                else: # 若非手脚，跳过
                    continue
                if 'sensor' in self.geomdict[con.geom1]+self.geomdict[con.geom2]:
                    # 更新当前的有效接触情况
                    limb_sensor_state[limb] = self.ladder_height[ladder]
            else: # 若碰撞对中不含阶梯，跳过
                continue
            cont_pair = (limb,ladder)    
            # 若当前碰到的阶梯高度比先前碰到的要低，倒扣分
            if self.ladder_height[ladder] < self.limb_position[limb]:
                if self.ladder_task != 0:
                    reward += -50
                    self.ladder_up = False
            # 更新肢体达到的最高位置
            self.limb_position[limb] = self.ladder_height[ladder] 

        if self.ladder_task == 0: # 0级任务 先把一只手放上，再放另一只手
            self._forward_speed_reward_weight = 0.2
            self._healthy_reward = 0.2
            
            if limb_sensor_state['right_hand'] == 5 and self.ladder_task_flag[0]['right_hand'] == False:
                # 右手成功触碰
                reward += 10
                self.ladder_task_flag[0]['right_hand'] = True
            if limb_sensor_state['right_hand'] != 5 and self.ladder_task_flag[0]['right_hand'] == True:
                reward -= 10
                self.ladder_task_flag[0]['right_hand'] = False
            if limb_sensor_state['left_hand'] == 5 and self.ladder_task_flag[0]['left_hand'] == False:
                # 左手成功触碰
                reward += 10
                self.ladder_task_flag[0]['left_hand'] = True
            if limb_sensor_state['left_hand'] != 5 and self.ladder_task_flag[0]['left_hand'] == True:
                reward -= 10
                self.ladder_task_flag[0]['left_hand'] = False                
            if limb_sensor_state['right_hand'] == 5 and limb_sensor_state['left_hand'] == 5 :
                reward += 5
        return reward
                    
    def _get_obs(self):
        # obs空间
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        com_inertia = self.sim.data.cinert.flat.copy()
        com_velocity = self.sim.data.cvel.flat.copy()

        actuator_forces = self.sim.data.qfrc_actuator.flat.copy()
        external_contact_forces = self.sim.data.cfrc_ext.flat.copy()

#        if self._exclude_current_positions_from_observation:
#            position = position[2:]
#        if self.terrain_type == 'steps':
#            position[0], position[2] = self._get_steps_pos()
        if self.terrain_type == 'default':
            position = position[2:]

        if self.terrain_type == 'steps':
            position = position[2:]
            steps_pos = self._get_steps_pos()
            position = np.append(position,steps_pos)

        if self.terrain_type == 'ladders':
            position = position[3:]
            ladders_pos = self._get_ladders_pos()
            position = np.append(position,ladders_pos)

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

    def _get_steps_pos(self):
        # 读取当前机器人相对于台阶的x,y,z距离: 
        step_size_x, step_size_y, step_size_z = self.xml_model.step_size
        x_w = self.sim.data.qpos[0] + step_size_x + 0.0001  # 机器人在全局的x坐标
        x_list=[]
        for i in range(7):
            x_ = x_w - 0.1 + i*0.1
            z_ = self.sim.data.qpos[2] - (x_ // step_size_x)*step_size_z # 机器人当前的距离地面（台阶）的高度
            x_list.append(z_)
        return x_list 

    def _get_ladders_pos(self):
        # 读取当前机器人相对于阶梯的x,y,z距离: 
        ladders_pos = self.xml_model.ladder_positions
        lowest_ladder = self.limb_position['left_foot'] if self.limb_position['left_foot'] < self.limb_position['right_foot'] else self.limb_position['right_foot']
        x_w = self.sim.data.qpos[0]  # 机器人在全局的x坐标
        z_w = self.sim.data.qpos[2]  # 机器人在全局的z坐标
        pos_list=[]
        for i in range(3):
            _x = ladders_pos[lowest_ladder+i][0] - x_w
            _z = - ladders_pos[lowest_ladder+i][2] + z_w
            pos_list.append(_x)
            pos_list.append(_z)
        right_hand_target_dis_x = self.sim.data.geom_xpos[41][0] - ladders_pos[4][0]   # right hand sensor position
        right_hand_target_dis_z = self.sim.data.geom_xpos[41][2] - ladders_pos[4][2]   # right hand sensor position
        left_hand_target_dis_x = self.sim.data.geom_xpos[45][0] - ladders_pos[4][0]   # right hand sensor position
        left_hand_target_dis_z = self.sim.data.geom_xpos[45][2] - ladders_pos[4][2]   # right hand sensor position

        return pos_list 

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
        xyz_velocity = (xyz_position_after[0:3] - xyz_position_before[0:3]) / self.dt
        x_velocity, y_velocity, z_velocity = xyz_velocity
        self.x_velocity = x_velocity
        self.z_velocity = z_velocity

        # 是否仍在前进，只调用一次，防止反复调用is_walking出错
        self._is_walking = self.is_walking

        # cost值 控制cost + 接触力cost
        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        # reward值 
        # _forward_reward_weight 默认为1.25, 将x轴位移添加到奖励值中，鼓励前进。
        forward_reward = self.forward_reward
        healthy_reward = self.healthy_reward
        posture_reward = self.posture_reward
        contact_reward = self.contact_reward

        rewards = forward_reward + healthy_reward + posture_reward + contact_reward
        costs = ctrl_cost + contact_cost

        observation = self._get_obs()
        reward = rewards - costs
        done = self.done
        self.reward_accumulate(forward_r=forward_reward,healthy_r=healthy_reward,posture_r=posture_reward,
                                contact_r=contact_reward,contact_c=contact_cost,control_c=ctrl_cost)
        info = {
            "reward_details":{"forward_reward_sum": self.forward_reward_sum,
                              "contact_reward_sum": self.contact_reward_sum,
                              "posture_reward_sum": self.posture_reward_sum,
                              "healthy_reward_sum": self.healthy_reward_sum,
                              "control_cost_sum": -self.control_cost_sum,
                              "contact_cost_sum": -self.contact_cost_sum,},
            "xyz_position": xyz_position_after,
            "distance_from_origin": np.linalg.norm(xyz_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
            "is_healthy": self.is_healthy,
            "is_walking": self._is_walking,
            "contact pairs":self.already_touched
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
        self.__init_counter()
        
        observation = self._get_obs()
        
        
        return observation
    
    def reward_accumulate(self, forward_r = 0, contact_r = 0, healthy_r = 0, posture_r = 0, control_c = 0, contact_c = 0):
        self.forward_reward_sum += forward_r
        self.contact_reward_sum += contact_r
        self.healthy_reward_sum += healthy_r
        self.posture_reward_sum += posture_r
        self.control_cost_sum += control_c
        self.contact_cost_sum += contact_c
        return 

    def viewer_setup(self):
        for key, value in self.camera_config.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)