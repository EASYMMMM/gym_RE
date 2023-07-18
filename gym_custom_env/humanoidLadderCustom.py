'''
humanoidLadderCustom.py
复制自 anaconda3\envs\GYM\Lib\site-packages\gym\envs\mujoco\humanoid_v3.py
在此基础上自定义mujoco ENV环境

humanoid 爬梯子的环境。专门用于梯子测试
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
    "distance": 8,   # 场地限制10：8.  场地限制15：11
    "lookat": np.array((2.5, 0.0, 1.0)), # FLAT FLOOR:[场地限制10：4. 场地限制13：6.3]
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

class HumanoidLadderCustomEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        xml_file:str =None, # assets文件夹下的xml文件
        terrain_type="ladders",
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
        terrain_info=True,
        reset_noise_scale=1e-2,
        camera_config = "horizontal",
        single_contact_reward = 10,
        env_gravity = -9.81,                                   # 环境重力
        use_origin_model = False,                          # 是否使用原版模型，控制器试验时用
        flatfloor_size = 10,                               # 平地的长度限制。默认10，录制视频时可调至15
        y_limit = True,                                    # 平地的y轴限制。训练时开启，测试时关闭。
        exclude_current_positions_from_observation=False,  # Flase: 使obs空间包括躯干的x，y坐标; True: 不包括
    ):
        utils.EzPickle.__init__(**locals())

        # 变量初始化
        self._forward_speed_reward_weight = 1 if terrain_type == 'default' else forward_speed_reward_weight
        self._forward_distance_reward_weight = forward_distance_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight if terrain_type in 'default'+'steps' else 0.2*ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._contact_cost_range = contact_cost_range
        self._healthy_reward = healthy_reward  
        self._single_contact_reward = single_contact_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._posture_reward_weight = posture_reward_weight
        self._contact_reward_weight = contact_reward_weight
        self._terrain_info = terrain_info
        self._reset_noise_scale = reset_noise_scale
        self._use_origin_model = use_origin_model
        self._flatfloor_size = flatfloor_size
        self._env_gravity = env_gravity
        self._y_limit = y_limit
        self.camera_config = {
            "defalt":DEFAULT_CAMERA_CONFIG,
            "horizontal":HORIZONTAL_CAMERA_CONFIG,
        }[camera_config]
        self.first_call = True
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        # set the terrain, generate XML file
        self.terrain_type = terrain_type        
        terrain_list = ('default','ladders') # 默认平地，台阶，梯子
        assert self.terrain_type in terrain_list, 'ERROR:Undefined terrain type'  
        xml_name = 'humanoid_ladder_exp.xml'
        self.xml_model = HumanoidXML(terrain_type=self.terrain_type, gravity=self._env_gravity)
        self.xml_model.write_xml(file_path=f"gym_custom_env/assets/{xml_name}")
        dir_path = os.path.dirname(__file__)
        xml_file_path = f"{dir_path}\\assets\\{xml_name}"


        self.__init_counter()
        if self._use_origin_model:
            xml_file_path = f"{dir_path}\\assets\\humanoid_origin.xml"
        print("============ HUMANOID LADDER CUSTOM ENV ============")
        print(f"=====terrain type:{self.terrain_type}=====")
        mujoco_env.MujocoEnv.__init__(self, xml_file_path, 5)
        self.geomdict = self.get_geom_idname()      # geom id name字典

    def __init_counter(self):
        # 初始化/清零需要用到的储存数组
        self._success   = False
        self.x_velocity = 0                         # 质心沿x速度
        self.y_velocity = 0                         # 质心沿y速度
        self.z_velocity = 0                         # 质心沿z速度
        self._walking_counter = 0                   # 判定正常前进计数器
        self.v_list = list()
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
        self.ladder_task_flag = { 0:{'right_hand':False, 'left_hand':False, 'last_right_hand_dis_r':0,'last_left_hand_dis_r':0} # 分解任务中用到的标志
                                    }  
        
    @property
    def is_walking(self):
        # 判断机器人是否正常运动，若连续200步速度小于某值，停止
        _is_walking = True
        if self.terrain_type == 'ladders':
            threshold = 200
        else:
            threshold = 100

        if self.z_velocity < 0.1:  #选取沿着z轴方向的运动
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

        # 梯子地形中，生存奖励设定为负数
        return (
            ( float((self.is_healthy or self._terminate_when_unhealthy) and  self._is_walking ) )
            * self._healthy_reward
        )
    
    @property
    def forward_reward(self):
        # 计算前进奖励    
        # 梯子地形
        # 前进奖励 = 速度权重*前进速度 
        if self.terrain_type == 'ladders':
            forward_reward = self._forward_speed_reward_weight * (self.x_velocity + 2*self.z_velocity) 
        return forward_reward

    @property
    def posture_reward(self):
        # 姿态奖励 = 直立 + 方向朝前
        reward = 0
        quatanion = self.sim.data.qpos[3:7]
        Rm = R.from_quat(quatanion)  # Rotation matrix
        rotation_matrix = Rm.as_matrix()
        # 提取躯干的z轴方向向量
        vertical_direction = np.array([0, 0, 1])
        body_z_axis = rotation_matrix.dot(vertical_direction)
        # 计算z轴方向向量与竖直向上方向向量的点积
        z_dot_product = np.dot(body_z_axis, vertical_direction)

        # 航向角计算
        # 提取躯干的x轴方向向量
        forward_direction = np.array([1, 0, 0])
        body_x_axis = rotation_matrix.dot(forward_direction)
        # 计算x轴方向向量与竖直向上方向向量的点积
        yaw = np.dot(body_x_axis, forward_direction)
        # 5-3更新 航向角reward使用y坐标表示
        y = self.sim.data.qpos[1]
        yaw = - 1.5 * y*y + 0.5

        # 将点积映射到[0, 1]范围内的奖励值
        # 楼梯地形同时考虑姿态和朝向
        if self.terrain_type == 'default':
            v_y = self.y_velocity if self.y_velocity > 0 else -self.y_velocity # 绝对值
            r_y = 0 if v_y < 0.4 else -(v_y - 0.4)
            #yaw = - 1.5 * y*y + 1
            #if yaw < -1:
            #    yaw = -1
            reward = self._posture_reward_weight *(r_y)
        # 阶梯地形为了防止持续获得奖励，朝向偏差太多时扣分，正向不额外给分。
        if self.terrain_type == "ladders":
            reward_x = 0 if self._posture_reward_weight*(( yaw + 1.0) / 2.0) > 0.8 else -1
            reward_z = 0 if self._posture_reward_weight*(( - z_dot_product + 1.0) / 2.0) > 0.8 else -1
            reward = reward_x + reward_z
            # FIXME :此处posture reward暂时设为0
            reward = 0
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
        # 对于steps，始终为True
        # 返回False：需要提前终止

        contact = list(self.sim.data.contact)  # 读取一个元素为mjContact的结构体数组
        ncon = self.sim.data.ncon # 碰撞对的个数
        not_fallen = True
        if self.terrain_type == 'ladders':
            # FIXME: 通过接触来判断摔倒
            return True
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
        return fallen
                
    @property
    def is_healthy(self):
        # 机器人状态是否正常，通过qpos[2]的z轴位置判断（是否跌倒）
        min_z, max_z = self._healthy_z_range
        z = self.sim.data.qpos[2]
        if self.terrain_type == 'default':
            x_limit = self.sim.data.qpos[0] < self._flatfloor_size # 别走出地图。 
            if self.sim.data.qpos[0] >= self._flatfloor_size:
                self._success = True
            if self._y_limit:
                y_limit = self.sim.data.qpos[1]<1 and self.sim.data.qpos[1]>-1 # 沿y轴也做出限制 
            else:
                y_limit = True
            is_inthemap = x_limit and y_limit
        if self.terrain_type == 'ladders':
            min_z = 0.9
            lowest_ladder = self.limb_position['left_foot'] if self.limb_position['left_foot'] < self.limb_position['right_foot'] else self.limb_position['right_foot']
            lowest_ladder_height = lowest_ladder * self.xml_model.ladder_positions[0][2]
            z = self.sim.data.qpos[2] - lowest_ladder_height
            # FIXME is in the map?
            is_inthemap = self.sim.data.qpos[2] < 3 and self.sim.data.qpos[0] > -1
            # if self.limb_position['right_hand'] == 11 or self.limb_position['left_hand'] == 11:  # 判断是否爬到最高的梯子
            #     is_inthemap = False
            # else:
            #     is_inthemap = True    

        is_standing = min_z < z < 10  #  self.sim.data.qpos[2]: z-coordinate of the torso (centre)
        is_healthy  = is_standing and is_inthemap and self._not_fallen

        if self.first_call :  # 更新了参数后，有时会出现初始化失败的现象。暴力解决。
            is_healthy = True
            self.first_call = False
        
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
        (3) moveLHand:  lift left hand to the next higher rung. 
        (4) moveRHand:  lift right hand to the next higher rung. 
        (6) moveLFoot:  lift left foot to the next higher rung. 
        (7) moveRFoot:  lift right foot to the next higher rung. 
        将有效碰撞体积限制为手掌、脚掌中心。
        ''' 
        self._forward_speed_reward_weight = 0.5
        # self._healthy_reward = 0.2
        contact = list(self.sim.data.contact)  # 读取一个元素为mjContact的结构体数组
        ncon = self.sim.data.ncon # 碰撞对的个数
        reward = 0
        self.limb_sensor_state = {'right_hand':100,    # 当前的肢体末端的有效接触情况，100表示悬空
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
                    self.limb_sensor_state[limb] = self.ladder_height[ladder]
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
            self._healthy_reward = 0.02
            ladders_pos = self.xml_model.ladder_positions
            # 鼓励把手从上方接近梯子
            right_hand_target_dis_x = - self.sim.data.geom_xpos[41][0] + ladders_pos[4][0]   # right hand sensor position
            right_hand_target_dis_z = self.sim.data.geom_xpos[41][2] - ladders_pos[4][2]   # right hand sensor position
            left_hand_target_dis_x = - self.sim.data.geom_xpos[45][0] + ladders_pos[4][0]   # right hand sensor position
            left_hand_target_dis_z = self.sim.data.geom_xpos[45][2] - ladders_pos[4][2]   # right hand sensor position
            right_hand_dis_r = 0
            left_hand_dis_r  = 0
            if right_hand_target_dis_z > 0.002:
                right_hand_dis_r = np.clip(0.5 - right_hand_target_dis_x,0,0.5) * 4
            if left_hand_target_dis_z > 0.002:
                left_hand_dis_r = np.clip(0.5 - left_hand_target_dis_x,0,0.5) * 4      
            reward = reward - self.ladder_task_flag[0]['last_right_hand_dis_r'] - self.ladder_task_flag[0]['last_left_hand_dis_r'] + right_hand_dis_r + left_hand_dis_r
            self.ladder_task_flag[0]['last_right_hand_dis_r'] = right_hand_dis_r
            self.ladder_task_flag[0]['last_left_hand_dis_r'] = left_hand_dis_r

            if self.limb_sensor_state['right_hand'] == 5 and self.ladder_task_flag[0]['right_hand'] == False:
                # 右手成功触碰
                reward += 10
                self.ladder_task_flag[0]['right_hand'] = True
            if self.limb_sensor_state['right_hand'] != 5 and self.ladder_task_flag[0]['right_hand'] == True:
                reward -= 10
                self.ladder_task_flag[0]['right_hand'] = False
            if self.limb_sensor_state['left_hand'] == 5 and self.ladder_task_flag[0]['left_hand'] == False:
                # 左手成功触碰
                reward += 10
                self.ladder_task_flag[0]['left_hand'] = True
            if self.limb_sensor_state['left_hand'] != 5 and self.ladder_task_flag[0]['left_hand'] == True:
                reward -= 10
                self.ladder_task_flag[0]['left_hand'] = False                
            if self.limb_sensor_state['right_hand'] == 5 and self.limb_sensor_state['left_hand'] == 5 :
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

        if self.terrain_type == 'default':
            position = position[2:] 

        if self.terrain_type == 'ladders':
            position = position[3:]
            ladders_pos = self._get_ladders_pos()
            position = np.append(position,ladders_pos)

        #print(f'position:{len(position)}')
        #print(f'velocity:{len(velocity)}')
        #print(f'com_inertia:{len(com_inertia)}')
        #print(f'com_velocity:{len(com_velocity)}')
        #print(f'actuator_forces:{len(actuator_forces)}')
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

    def _ladder_hand_distance(self):
        # TODO 手部sensor到目标梯子的距离
        # 如果单纯计算速度作为reward，很难保证绕开梯子抓住
        pass

    def _get_steps_pos(self, geom_name:str = 'pelvis_geom'):
        # 读取当前机器人某个部位相对于台阶的x,y,z距离: 
        geom_x, geom_y, geom_z = self.sim.data.get_geom_xpos(geom_name) # 通过get_geom_xpos获得几何体的坐标
        step_size_x, step_size_y, step_size_z = self.xml_model.step_size
        x_w = geom_x + step_size_x + 0.0001  # 机器人在全局的x坐标
        height_list=[]        # 一系列坐标点到楼梯的高度
        for i in range(7):
            x_ = x_w - 0.1 + i*0.1
            z_ = geom_z - (x_ // step_size_x)*step_size_z # 机器人当前的距离地面（台阶）的高度
            height_list.append(z_)
        return height_list 

    '''
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
        pos_list.append(right_hand_target_dis_x)
        right_hand_target_dis_z = self.sim.data.geom_xpos[41][2] - ladders_pos[4][2]   # right hand sensor position
        pos_list.append(right_hand_target_dis_z)
        left_hand_target_dis_x = self.sim.data.geom_xpos[45][0] - ladders_pos[4][0]   # right hand sensor position
        pos_list.append(left_hand_target_dis_x)
        left_hand_target_dis_z = self.sim.data.geom_xpos[45][2] - ladders_pos[4][2]   # right hand sensor position
        pos_list.append(left_hand_target_dis_z)
        for i in self.limb_sensor_state.values():
            if i < 100:
                pos_list.append(1) # 肢体末端与地面接触
            else:
                pos_list.append(0) # 肢体末端与地面接触   
        return pos_list 
    '''

    def _get_ladders_pos(self):
        # 读取当前机器人相对于阶梯的x,y,z距离:
        # 获取全部阶梯的位置
        ladders_pos = self.xml_model.ladder_positions
        lowest_ladder = self.limb_position['left_foot'] if self.limb_position['left_foot'] < self.limb_position['right_foot'] else self.limb_position['right_foot']
        x_w = self.sim.data.qpos[0]  # 机器人在全局的x坐标
        z_w = self.sim.data.qpos[2]  # 机器人在全局的z坐标
        lowest_ladder_num = 0
        for l_p in ladders_pos: # 找到机器人能触及的范围内最低的一层阶梯 （torso往下1.4）
            if l_p[2] > z_w - 1.45:
                lowest_ladder = l_p
                break
            lowest_ladder_num += 1
        
        # 扫描从最低往上共9个横杆，读取torso相对于横杆的数据
        pos_list=[]
        for i in range(9):
            _x = ladders_pos[lowest_ladder_num+i][0] - x_w
            _z = - ladders_pos[lowest_ladder_num+i][2] + z_w
            pos_list.append(_x)
            pos_list.append(_z)
        # 扫描9个横杆中最上方的4个横杆，读取左右手相对于横杆的数据
        for i in range(4):
            right_hand_target_dis_x = self.sim.data.get_geom_xpos('right_hand_sensor_geom')[0] - ladders_pos[lowest_ladder_num+5+i][0]
            pos_list.append(right_hand_target_dis_x)
            right_hand_target_dis_z = self.sim.data.get_geom_xpos('right_hand_sensor_geom')[2] - ladders_pos[lowest_ladder_num+5+i][2]
            pos_list.append(right_hand_target_dis_z)
            left_hand_target_dis_x = self.sim.data.get_geom_xpos('left_hand_sensor_geom')[0] - ladders_pos[lowest_ladder_num+5+i][0]
            pos_list.append(left_hand_target_dis_x)
            left_hand_target_dis_z = self.sim.data.get_geom_xpos('left_hand_sensor_geom')[2] - ladders_pos[lowest_ladder_num+5+i][2]
            pos_list.append(left_hand_target_dis_z)
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

        # 进行一步的仿真
        xyz_position_before = mass_center(self.model, self.sim)        
        self.do_simulation(action, self.frame_skip)
        xyz_position_after = mass_center(self.model, self.sim)

        # dt为父类mujoco_env中的一个@property函数 
        xyz_velocity = (xyz_position_after[0:3] - xyz_position_before[0:3]) / self.dt
        x_velocity, y_velocity, z_velocity = xyz_velocity
        self.x_velocity = x_velocity
        self.y_velocity = y_velocity
        self.z_velocity = z_velocity
        self.v_list.append(x_velocity)
        
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
        # FIXME：contact reward暂时设为0
        # contact_reward = self.contact_reward
        contact_reward = 0

        rewards = forward_reward + healthy_reward + posture_reward + contact_reward
        costs = ctrl_cost + contact_cost

        observation = self._get_obs()
        reward = rewards - costs
        done = self.done
        self.reward_accumulate(forward_r=forward_reward,healthy_r=healthy_reward,posture_r=posture_reward,
                                contact_r=contact_reward,contact_c=contact_cost,control_c=ctrl_cost)

        v_ave = sum(self.v_list)/len(self.v_list)
        
        info = {
            "reward_details":{"forward_reward_sum": self.forward_reward_sum,
                              "contact_reward_sum": self.contact_reward_sum,
                              "posture_reward_sum": self.posture_reward_sum,
                              "healthy_reward_sum": self.healthy_reward_sum,
                              "control_cost_sum": -self.control_cost_sum,
                              "contact_cost_sum": -self.contact_cost_sum,
                              "final_x":xyz_position_after[0],
                              "ave_velocity":v_ave},
            "xyz_position": xyz_position_after,
            "distance_from_origin": np.linalg.norm(xyz_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
            "is_healthy": self.is_healthy,
            "is_walking": self._is_walking,
            "contact pairs":self.already_touched,
            "is_success":self._success,
            "ave_velocity":v_ave,
        }

        return observation, reward, done, info

    def update_xml_model(self,params):
        # 更新XML文档，更新仿真模型
        print('----- Update XML Model ----- ')
        xml_name = 'humanoid_ladder_exp.xml'
        self.xml_model.set_params(params)
        self.xml_model.update_xml(file_path=f"gym_custom_env/assets/{xml_name}")
        dir_path = os.path.dirname(__file__)
        xml_file_path = f"{dir_path}\\assets\\{xml_name}"
        #if params['thigh_lenth']+params['shin_lenth']<0.5:
        #    self._healthy_z_range = (0.5,5.0)
        #else:
        #    self._healthy_z_range = (0.8,5.0)
        mujoco_env.MujocoEnv.__init__(self, xml_file_path, 5)


    def reset_xml_model(self):
        print('----- Reset XML Model ----- ')
        xml_name = 'humanoid_ladder_exp.xml'
        self.xml_model.reset_params()
        self.xml_model.update_xml(file_path=f"gym_custom_env/assets/{xml_name}")
        dir_path = os.path.dirname(__file__)
        xml_file_path = f"{dir_path}\\assets\\{xml_name}"
        mujoco_env.MujocoEnv.__init__(self, xml_file_path, 5)        


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