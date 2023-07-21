import xml.etree.ElementTree as ET
from typing import Any, Dict, Optional, Type, Union, List
import sys,os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from xml_util import XMLPart,MujocoXML ,prettyXml
from math import sin,cos,pi

'''
    定义HumanoidXML类，继承XMLTree类，定义具有人形机器人的XML模型文档。
'''
class HumanoidXML(MujocoXML):
    '''
    生成基于Mujoco的具有Humanoid Robot的XML文档。
    可更改地形，更改Robot各个部件的尺寸。
    各个部件的连接方式不变。
    各个驱动joint不变。
    '''
    def __init__(self, 
                 root_tag:str = "mujoco",
                 terrain_type:str = "default",
                 gravity:float = -9.81,
                 ):
        super(HumanoidXML ,self).__init__(root_tag=root_tag,gravity=gravity)

        self.terrain_type = terrain_type
        self.gravity = gravity
        terrain_type_list = ('default','steps','ladders') # 默认平地，台阶，梯子
        assert self.terrain_type in terrain_type_list, 'ERROR:Undefined terrain type'
        init_pos = {'default':[-1,0,1.4] , 'steps':[-0.6,0,1.4], 'ladders':[-0.3,0,1.4]}[self.terrain_type] # ladders -0.3,0,1.4
        self.__default_param_list = { 'init_position':init_pos,
                            'head_radius' : 0.18,          # 头部半径 0.18
                            'torso_width': 0.14,           # 躯干宽 0.14
                            'torso_height': 0.425,         # 躯干高 0.425
                            'waist_lenth':0.12,            # 腰部宽 0.12
                            'pelvis_width':0.14,           # 骨盆宽 0.14
                            'thigh_lenth':0.34,            # 大腿长 0.34
                            'thigh_size':0.06,             # 大腿粗 0.06
                            'shin_lenth':0.3,              # 小腿长 0.3
                            'shin_size':0.05,              # 小腿粗 0.05
                            'upper_arm_lenth':0.2771,      # 大臂长 0.2771
                            'upper_arm_size':0.04,         # 大臂粗 0.04
                            'lower_arm_lenth':0.2944,      # 小臂长 0.2944
                            'lower_arm_size':0.031,        # 小臂粗 0.031
                            'foot_lenth':0.18,             # 脚长   0.18
                            'steps_height':0.10,           # 楼梯高 0.10
                            'gravity':-9.81,                # 重力 -9.81
                            }
        self.param_list = self.__default_param_list.copy()

    def _basic_structure(self):
        '''
        生成XML文档的基本框架
        '''
        self._add_compiler()
        self._add_default()
        self._add_option()
        self._add_size()
        self._add_visual()
        self._add_asset()
        self._add_worldbody()
    
    def _add_worldbody(self, 
                      born_position: List[Union[float,int]] = [0.0,0.0,1.4],
                      tag = "worldbody"
                      ):
        worldbody = self.root.child_element(tag,attributes={})
        self.elements.update({tag:worldbody})
        light_attr = {"cutoff":"100",
                      "diffuse":"1 1 1",
                      "dir":"-0 0 -1.3",
                      "directional":"true",
                      "exponent":"1",
                      "pos":"0 0 1.3",
                      "specular":".1 .1 .1",        } 
        worldbody_light = worldbody.child_element("light",light_attr)
        camera_attr_3 = {'name':'track_3',
                       'mode':'fixed',
                       'pos':'6.5 0 11.5',
                       'xyaxes':'1 0 0 0 1 0',}
        camera_3 = worldbody.child_element("camera",camera_attr_3)
        camera_attr_4 = {'name':'staris_camera',
                         'mode':'fixed',
                       'pos':'-1 -6 2.6',
                       'xyaxes':'0.88 -0.5 0 0.1 0.1 0.99',}
        '''
                camera_attr_4 = {'name':'staris_camera',
                         'mode':'fixed',
                       'pos':'-3 -5 2.6',
                       'xyaxes':'0.7 -0.7 0 0.1 0.1 0.99',}
                       '''
        camera_4 = worldbody.child_element("camera",camera_attr_4)


    def set_terrain(self,
                    name: Union[str, None] = None,
                    size: List[str] = None):
        '''
        定义地形，为worldbody的子元素
        '''

        terrain_type = self.terrain_type
        # 使用默认地板的地形种类
        default_floor_terrain_type = ('default','steps','ladders') # 默认平地，台阶，梯子
        tag = "geom"

        if terrain_type in default_floor_terrain_type:
            # 定义默认地板
            if not name:
                name = "flatfloor"
            if not size:
                Size = "20 20 2"
            else:
                Size = self.list2str(size)
            terrain_attr = {"condim": "3",
                            "friction": "1 .1 .1",
                            "material": "MatPlane",
                            "name": name,
                            "pos": "0 0 0",
                            "rgba": "0.8 0.9 0.8 1",
                            "size": Size,
                            "type": "plane", }
            self.elements["worldbody"].child_element(tag, terrain_attr)

        # 定义梯子
        if terrain_type == "ladders":
            # Define the parameters for the box geometries
            box_size =  [0.06,1.2,0.01]
            box_rgba = "0 .9 0 1"
            bottom_rgba = "0.9 0 0 1"
            box_condim = "3"
            box_friction = "1 .1 .1"

            # Define the positions using an arithmetic sequence with a common difference of 0.2
            # TODO: 梯子结构
            #      原版梯子间距：单个梯子宽0.12，梯子间距0.10，有重合。梯子高度0.3
            #      4.17修改：梯子间距0.18
            #      4.19修改：梯子间距0.16
            #      4.20修改：梯子间距0.10，高度0.25（按照论文
            #      7.18修改：梯子间距0.05
            positions = [(round((i) * 0.05, 3), 0, round((i) * 0.25, 3)) for i in range(16)]
            self.ladder_positions = positions
            # Create a geometry for each position
            for i, pos in enumerate(positions):
                box_attr = {
                    "name": f"ladder{i + 1}",
                    "type": "box",
                    "size": f"{box_size[0]*0.9} {box_size[1]} {box_size[2]*0.5}" ,
                    "pos": f"{pos[0]} {pos[1]} {pos[2] + box_size[2]*0.25}",
                    "rgba": box_rgba,
                    "condim": box_condim,
                    "friction": box_friction,
                }
                self.elements["worldbody"].child_element(tag, box_attr)
                bottom_attr = {
                    "name": f"bottom{i + 1}",
                    "type": "box",
                    "size": f"{box_size[0]} {box_size[1]} {box_size[2]*0.5}" ,
                    "pos": f"{pos[0]} {pos[1]} {pos[2] - box_size[2]*0.25}",
                    "rgba": bottom_rgba,
                    "condim": box_condim,
                    "friction": box_friction,
                }
                self.elements["worldbody"].child_element(tag, bottom_attr)                
        
        # 定义楼梯
        if terrain_type == 'steps':     
            # Define the parameters for the box geometries
            # should be equal to the xyz in box_size
            box_x = 0.3
            box_y = 1.2
            box_z = self.param_list['steps_height']    # 原版高度： 0.1
            box_size = f"{box_x} {box_y} {box_z}"
            self.step_size = [box_x*2,box_y*2,box_z*2]

            box_rgba = "0 .9 0 1"
            box_condim = "3"
            box_friction = "1 .1 .1"

            # Define the positions using an arithmetic sequence with a common difference of 0.2
            positions = [(round((i+1) * box_x+ (i) * box_x, 3), 0, round((i+1) * box_z+(i) * box_z, 3)) for i in range(11)]

            # Create a body element for the boxes,named "ground"

            # Create a geometry for each position
            for i, pos in enumerate(positions):
                box_attr = {
                    "type": "box",
                    "size": box_size,
                    "pos": f"{pos[0]} {pos[1]} {pos[2]}",
                    "rgba": box_rgba,
                    "condim": box_condim,
                    "friction": box_friction,
                    "name": f"step{i+1}"
                }
                self.elements["worldbody"].child_element(tag, box_attr)


    
    def _create_humanoid(self ) -> XMLPart:
        '''
        创建机器人
        先创建躯干，随后在躯干的基础上添加下半身、左臂、右臂
        '''
        worldbody = self.elements["worldbody"]

        torso_attr = {"name":"torso", "pos":self.list2str(self.param_list["init_position"])}
        torso = worldbody.child_element("body", torso_attr)  # 躯干
        camera_attr = {'name':'track',
                       'mode':'trackcom',
                       'pos':'0 -4 0',
                       'xyaxes':'1 0 0 0 0 1',}
        camera_1 = torso.child_element("camera",camera_attr)
        camera_attr_2 = {'name':'track_2',
                       'mode':'trackcom',
                       'pos':'0 -6 0',
                       'xyaxes':'1 0 0 0 0 1',}
        camera_2 = torso.child_element("camera",camera_attr_2)

        root_joint = self.add_joint( name='root',
                                     parent = torso,
                                     joint_type='free', 
                                     armature = 0 , 
                                     damping = 0, 
                                     limited = 'false',
                                     pos = [0,0,0],
                                     stiffness = 0 )
        # 保持躯干一体性，躯干整体等比例放大                             
        torso_width_scale = self.param_list['torso_width']/self.__default_param_list['torso_width']
        torso_height_scale = self.param_list['torso_height']/self.__default_param_list['torso_height']
        # 躯干（肩部）
        torso_geom = self.add_geom(name = 'torso_geom',
                                   parent = torso,
                                   geom_type = 'capsule',
                                   from_point = [0, -self.param_list['torso_width']/2, 0],
                                   to_point = [0, self.param_list['torso_width']/2, 0],
                                   size = 0.07 )
        
        head_geom = self.add_geom(name = 'head_geom',
                                  parent = torso,
                                  geom_type = 'sphere',
                                  pos = [0, 0, self.param_list['head_radius']*1.1],
                                  size = self.param_list['head_radius']/2,
                                  user = 258)                            
        uwaist_geom = self.add_geom(name='uwaist_geom',
                                    parent = torso,
                                    size = 0.06,
                                    geom_type = 'capsule',
                                    from_point= [-0.01, -torso_width_scale*self.__default_param_list['waist_lenth']/2, -0.12*torso_height_scale],
                                    to_point= [-0.01, torso_width_scale*self.__default_param_list['waist_lenth']/2, -0.12*torso_height_scale])
        # 创建下半身
        self._create_lowerbody(torso)
        self._create_arms(torso)
        return torso

    def _create_lowerbody(self, torso:XMLPart) ->XMLPart:
        '''
        创建下半身
        '''
        # 保持躯干一体性，躯干整体等比例放大                             
        torso_width_scale = self.param_list['torso_width']/self.__default_param_list['torso_width'] 
        torso_height_scale = self.param_list['torso_height']/self.__default_param_list['torso_height']  
        lwaist_z_pos = 0.260*torso_height_scale
        lwaist_attr = { 'name':'lwaist',
                        'pos' : f'-.01 0 -{lwaist_z_pos}',
                        'quat': "1.000 0 -0.002 0"}
        lowerbody = torso.child_element('body',lwaist_attr)
        lwaist_geom = self.add_geom(name='lwaist_geom',
                                    parent=lowerbody,
                                    geom_type='capsule',
                                    from_point=[0,-torso_width_scale*self.__default_param_list['waist_lenth']/2,0],
                                    to_point=[0,torso_width_scale*self.__default_param_list['waist_lenth']/2,0],
                                    size=0.06,
                                    )
        abdomen_z_joint = self.add_joint(name='abdomen_z',
                                         parent=lowerbody,
                                         joint_type='hinge',
                                         armature=0.02,
                                         axis=[0,0,1],
                                         damping=5,
                                         pos=[0,0,0.065],
                                         joint_range=[-45,45],
                                         stiffness=20  )
        abdomen_y_joint = self.add_joint(name='abdomen_y',
                                         parent=lowerbody,
                                         joint_type='hinge',
                                         armature=0.02,
                                         axis=[0,1,0],
                                         damping=5,
                                         pos=[0,0,0.065],
                                         joint_range=[-75,30],
                                         stiffness=10  )    
        self._create_pelvis(lowerbody)
        return lowerbody

    def _create_pelvis(self,lowerbody:XMLPart) ->XMLPart:
        '''
        创建骨盆
        '''
        torso_height_scale = self.param_list['torso_height']/self.__default_param_list['torso_height'] 
        pelvis_pos = 0.165* torso_height_scale
        pelvis_attr = { 'name':'pelvis',
                        'pos':f'0 0 -{pelvis_pos}',
                        'quat':'1.000 0 -0.002 0'}
        pelvis = lowerbody.child_element('body',pelvis_attr)               
        abdomen_x_joint = self.add_joint(name='abdomen_x',
                                         parent=pelvis,
                                         joint_type='hinge',
                                         armature=0.02,
                                         axis=[1,0,0],
                                         damping=5,
                                         pos=[0,0,0.1],
                                         joint_range=[-35,35],
                                         stiffness=10,       )
        pelvis_geom = self.add_geom(name='pelvis_geom',
                                    parent=pelvis,
                                    geom_type='capsule',
                                    from_point=[-0.02,-self.param_list['pelvis_width']/2,0],
                                    to_point=[-0.02,self.param_list['pelvis_width']/2,0],
                                    size=0.09)
        self._create_legs(pelvis)
        return pelvis

    def _create_legs(self, pelvis:XMLPart):
        '''
        生成左腿和右腿
        '''
        # 右腿
        leg_pos = self.param_list['pelvis_width']/2 + 0.03
        right_thigh_attr = { 'name':'right_thigh',
                             'pos':f'0 -{leg_pos} -0.04'}
        right_thigh = pelvis.child_element('body',right_thigh_attr)
        right_hip_x_joint = self.add_joint(name='right_hip_x',
                                           parent=right_thigh,
                                           joint_type='hinge',
                                           armature=0.01,
                                           axis=[1,0,0],
                                           damping=5,
                                           pos=[0,0,0],
                                           joint_range=[-25,5],
                                           stiffness=10)
        right_hip_z_joint = self.add_joint(name='right_hip_z',
                                           parent=right_thigh,
                                           joint_type='hinge',
                                           armature=0.01,
                                           axis=[0,0,1],
                                           damping=5,
                                           pos=[0,0,0],
                                           joint_range=[-60,35],
                                           stiffness=10)
        right_hip_y_joint = self.add_joint(name='right_hip_y',
                                           parent=right_thigh,
                                           joint_type='hinge',
                                           armature=0.01,
                                           axis=[0,1,0],
                                           damping=5,
                                           pos=[0,0,0],
                                           joint_range=[-110,20],
                                           stiffness=20)
        right_thigh_geom = self.add_geom(name='right_thigh_geom',
                                         parent=right_thigh,
                                         geom_type='capsule',
                                         from_point=[0,0,0],
                                         to_point=[0, 0.01, -self.param_list['thigh_lenth']],
                                         size=self.param_list['thigh_size']
                                         )
        shin_pos = self.param_list['thigh_lenth']+self.param_list['thigh_size']+0.003
        right_shin_attr = { 'name':'right_shin',
                             'pos':f'0 0.01 -{shin_pos}'}
        right_shin = right_thigh.child_element('body',right_shin_attr)
        right_knee_joint = self.add_joint(name='right_knee',
                                          parent=right_shin,
                                          joint_type='hinge',
                                          armature=0.006,
                                          axis=[0,-1,0],
                                          pos=[0,0,0.02],
                                          joint_range=[-160,-2] )
        right_shin_geom = self.add_geom(name='right_shin_geom',
                                        parent=right_shin,
                                        geom_type='capsule',
                                        from_point=[0,0,0],
                                        to_point=[0,0,-self.param_list['shin_lenth']],
                                        size=self.param_list['shin_size']
                                        ) 
        foot_pos = self.param_list['shin_lenth']+self.param_list['shin_size']+0.03
        # 右脚
        right_foot_attr = {'name':'right_foot', 'pos':f'0 0 -{foot_pos}'}  
        right_foot = right_shin.child_element('body',right_foot_attr)
        right_ankle_joint = self.add_joint(name='right_ankle',
                                          parent=right_foot,
                                          joint_type='hinge',
                                          armature=0.006,
                                          axis=[0,1,0],
                                          pos=[0,0,0],
                                          joint_range=[-45,25] )        
        right_foot_geom_1 = self.add_geom(name='right_foot_geom_1',
                                        parent=right_foot,
                                        geom_type='capsule',
                                        from_point=[-0.03,0,0],
                                        to_point=[-0.03+self.param_list['foot_lenth']*cos(8/180*pi),self.param_list['foot_lenth']*sin(8/180*pi),0],
                                        size=0.025,
                                        user=0)     
        right_foot_geom_2 = self.add_geom(name='right_foot_geom_2',
                                        parent=right_foot,
                                        geom_type='capsule',
                                        from_point=[-0.03,0,0],
                                        to_point=[-0.03+self.param_list['foot_lenth']*cos(8/180*pi),-self.param_list['foot_lenth']*sin(8/180*pi),0],
                                        size=0.025,
                                        user=0)       
        if self.terrain_type == 'ladders':
            right_foot_geom_3 = self.add_geom(name='right_foot_sensor_geom',
                                            parent=right_foot,
                                            geom_type='capsule',
                                            from_point=[-0.03+0.18*cos(8/180*pi)/2,-0.18*sin(8/180*pi)/2,0],
                                            to_point=[-0.03+0.18*cos(8/180*pi)/2,0.18*sin(8/180*pi)/2,0],
                                            size=0.0252,
                                            rgba=[1,0,0,1],
                                            user=0)                                                                                        

        # 左腿
        left_thigh_attr = { 'name':'left_thigh',
                             'pos':f'0 {leg_pos} -0.04'}
        left_thigh = pelvis.child_element('body',left_thigh_attr)
        left_hip_x_joint  = self.add_joint(name='left_hip_x',
                                           parent=left_thigh,
                                           joint_type='hinge',
                                           armature=0.01,
                                           axis=[-1,0,0],
                                           damping=5,
                                           pos=[0,0,0],
                                           joint_range=[-25,5],
                                           stiffness=10)
        left_hip_z_joint  = self.add_joint(name='left_hip_z',
                                           parent=left_thigh,
                                           joint_type='hinge',
                                           armature=0.01,
                                           axis=[0,0,-1],
                                           damping=5,
                                           pos=[0,0,0],
                                           joint_range=[-60,35],
                                           stiffness=10)        
        left_hip_y_joint  = self.add_joint(name='left_hip_y',
                                           parent=left_thigh,
                                           joint_type='hinge',
                                           armature=0.01,
                                           axis=[0,1,0],
                                           damping=5,
                                           pos=[0,0,0],
                                           joint_range=[-110,20],
                                           stiffness=20)

        left_thigh_geom  = self.add_geom(name='left_thigh_geom',
                                         parent=left_thigh,
                                         geom_type='capsule',
                                         from_point=[0,0,0],
                                         to_point=[0, -0.01, -self.param_list['thigh_lenth']],
                                         size=self.param_list['thigh_size']
                                         )
        left_shin_attr  =  { 'name':'left_shin',
                             'pos':f'0 -0.01 -{shin_pos}'}
        left_shin = left_thigh.child_element('body',left_shin_attr)
        left_knee_joint  = self.add_joint(name='left_knee',
                                          parent=left_shin,
                                          joint_type='hinge',
                                          armature=0.006,
                                          axis=[0,-1,0],
                                          pos=[0,0,0.02],
                                          joint_range=[-160,-2] )
        left_shin_geom  = self.add_geom(name='left_shin_geom',
                                        parent=left_shin,
                                        geom_type='capsule',
                                        from_point=[0,0,0],
                                        to_point=[0,0,-self.param_list['shin_lenth']],
                                        size=self.param_list['shin_size']
                                        ) 
        left_foot_attr  = {'name':'left_foot', 'pos':f'0 0 -{foot_pos}'}  
        # 左脚
        left_foot = left_shin.child_element('body',left_foot_attr)
        left_ankle_joint = self.add_joint(name='left_ankle',
                                          parent=left_foot,
                                          joint_type='hinge',
                                          armature=0.006,
                                          axis=[0,1,0],
                                          pos=[0,0,0],
                                          joint_range=[-45,25] )  
        left_foot_geom_1  = self.add_geom(name='left_foot_geom_1',
                                        parent=left_foot,
                                        geom_type='capsule',
                                        from_point=[-0.03,0,0],
                                        to_point=[-0.03+self.param_list['foot_lenth']*cos(8/180*pi),self.param_list['foot_lenth']*sin(8/180*pi),0],
                                        size=0.025,
                                        user=0)                  
        left_foot_geom_2 = self.add_geom(name='left_foot_geom_2',
                                        parent=left_foot,
                                        geom_type='capsule',
                                        from_point=[-0.03,0,0],
                                        to_point=[-0.03+self.param_list['foot_lenth']*cos(8/180*pi),-self.param_list['foot_lenth']*sin(8/180*pi),0],
                                        size=0.025,
                                        user=0)   
        if self.terrain_type == 'ladders':
            left_foot_geom_3 = self.add_geom(name='left_foot_sensor_geom',
                                            parent=left_foot,
                                            geom_type='capsule',
                                            from_point=[-0.03+0.18*cos(8/180*pi)/2,-0.18*sin(8/180*pi)/2,0],
                                            to_point=[-0.03+0.18*cos(8/180*pi)/2,0.18*sin(8/180*pi)/2,0],
                                            size=0.0252,
                                            rgba=[1,0,0,1],
                                            user=0)  

    def _create_arms(self, torso:XMLPart):
        '''
        生成手臂
        '''
        # 右臂
        shoulder_pos = self.param_list['torso_width']/2 + 0.1
        right_upper_arm_attr = { 'name':'right_upper_arm','pos':f'0 -{shoulder_pos} 0.06'}
        right_upper_arm = torso.child_element('body',right_upper_arm_attr)
        right_shoulder1_joint = self.add_joint(name='right_shoulder1',
                                               parent=right_upper_arm,
                                               joint_type='hinge',
                                               armature=0.0068,
                                               axis=[0,1,0],
                                               pos=[0,0,0],
                                               joint_range=[-95,80],
                                               stiffness=1)
        right_shoulder2_joint = self.add_joint(name='right_shoulder2',
                                               parent=right_upper_arm,
                                               joint_type='hinge',
                                               armature=0.0051,
                                               axis=[0,0,1],
                                               pos=[0,0,0],
                                               joint_range=[-85,30],
                                               stiffness=1)
        upper_arm_pos = pow(self.param_list['upper_arm_lenth']*self.param_list['upper_arm_lenth']/3,0.5)                                       
        right_upper_arm_geom = self.add_geom(name='right_upper_arm_geom',
                                             parent=right_upper_arm,
                                             geom_type='capsule',
                                             from_point=[0,0,0],
                                             to_point=[self.param_list['upper_arm_lenth'],0,0],
                                             size=self.param_list['upper_arm_size'])
        lower_arm_begin_pos = self.param_list['upper_arm_lenth']+self.param_list["upper_arm_size"]/2
        right_lower_arm_attr = {'name':'right_lower_arm', 'pos':f'{lower_arm_begin_pos} 0 0' }   
        right_lower_arm = right_upper_arm.child_element('body',right_lower_arm_attr)
        right_elbow_joint = self.add_joint( name='right_elbow',
                                            parent=right_lower_arm,
                                            joint_type='hinge',
                                            armature=0.0028,
                                            axis=[0,1,0],
                                            pos=[0,0,0],
                                            joint_range=[-50,90],
                                            stiffness=0)
        lower_arm_pos = pow(self.param_list['lower_arm_lenth']*self.param_list['lower_arm_lenth']/3,0.5) 
        right_lower_arm_geom = self.add_geom(name='right_lower_arm_geom',
                                             parent=right_lower_arm,
                                             geom_type='capsule',
                                             from_point=[0, 0, 0],
                                             to_point=[0,0,self.param_list['lower_arm_lenth']],
                                             size=self.param_list['lower_arm_size'])
     
        # 左臂
        left_upper_arm_attr = { 'name':'left_upper_arm','pos':f'0 {shoulder_pos} 0.06'}
        left_upper_arm = torso.child_element('body',left_upper_arm_attr)
        left_shoulder1_joint = self.add_joint(name='left_shoulder1',
                                               parent=left_upper_arm,
                                               joint_type='hinge',
                                               armature=0.0068,
                                               axis=[0,1,0],
                                               pos=[0,0,0],
                                               joint_range=[-95,80],
                                               stiffness=1)
        left_shoulder2_joint = self.add_joint(name='left_shoulder2',
                                               parent=left_upper_arm,
                                               joint_type='hinge',
                                               armature=0.0051,
                                               axis=[0,0,1],
                                               pos=[0,0,0],
                                               joint_range=[-30,85],
                                               stiffness=1)
        left_upper_arm_geom = self.add_geom(name='left_upper_arm_geom',
                                             parent=left_upper_arm,
                                             geom_type='capsule',
                                             from_point=[0,0,0],
                                             to_point=[self.param_list['upper_arm_lenth'],0,0],
                                             size=self.param_list['upper_arm_size'])
        left_lower_arm_attr = {'name':'left_lower_arm', 'pos':f'{lower_arm_begin_pos} 0 0' }   
        left_lower_arm = left_upper_arm.child_element('body',left_lower_arm_attr)
        left_elbow_joint = self.add_joint( name='left_elbow',
                                            parent=left_lower_arm,
                                            joint_type='hinge',
                                            armature=0.0028,
                                            axis=[0,1,0],
                                            pos=[0,0,0],
                                            joint_range=[-50,90],
                                            stiffness=0)
        left_lower_arm_geom = self.add_geom(name='left_lower_arm_geom',
                                             parent=left_lower_arm,
                                             geom_type='capsule',
                                             from_point=[0, 0, 0],
                                             to_point=[0,0,self.param_list['lower_arm_lenth']],
                                             size=self.param_list['lower_arm_size'])
        if self.terrain_type == 'steps':
            # 如果是阶梯地形，手部用球体表示
            right_hand_geom = self.add_geom(name='right_hand',
                                            parent=right_lower_arm,
                                            geom_type='sphere',
                                            pos=[0,0,self.param_list['lower_arm_lenth']],
                                            size=self.param_list['lower_arm_size']*1.2)              
            left_hand_geom = self.add_geom(name='left_hand',
                                            parent=left_lower_arm,
                                            geom_type='sphere',
                                            pos=[0,0,self.param_list['lower_arm_lenth']],
                                            size=self.param_list['lower_arm_size']*1.2)     
                                       
        if self.terrain_type == 'ladders':
            # 如果是阶梯地形，手部用球体表示
            hand_begin_poiot = self.param_list['lower_arm_lenth']
            right_hand_attr = {'name':'right_hand', 'pos':f'0 0 {self.param_list["lower_arm_lenth"]}'}   
            right_hand = right_lower_arm.child_element('body',right_hand_attr)                                                     
            right_hand_geom_1 = self.add_geom(name='right_hand_geom',
                                            parent=right_hand,
                                            geom_type='capsule',
                                            from_point=[0+0.02,0,0],
                                            to_point=[0.15+0.02,0,0],
                                            size=self.param_list['lower_arm_size']) 
            right_hand_geom_2 = self.add_geom(name='right_hand_sensor_geom',
                                            parent=right_hand,
                                            geom_type='capsule',
                                            from_point=[0.15/2-0.002 +0.02,0,0],
                                            to_point=[0.15/2+0.002 +0.02,0,0],
                                            size=self.param_list['lower_arm_size']+0.002,
                                            rgba=[1,0,0,1]) 
            right_wrist_joint = self.add_joint( name='right_wrist',
                                                parent=right_hand,
                                                joint_type='hinge',
                                                armature=0.0028,
                                                axis=[0,1,0],
                                                pos=[0,0,0],
                                                joint_range=[-80,20],
                                                stiffness=0)       

            left_hand_attr = {'name':'left_hand', 'pos':f'0 0 {self.param_list["lower_arm_lenth"]}'}   
            left_hand = left_lower_arm.child_element('body',left_hand_attr)                                                    
            left_hand_geom_1 = self.add_geom(name='left_hand_geom',
                                            parent=left_hand,
                                            geom_type='capsule',
                                            from_point=[0 +0.02,0,0],
                                            to_point=[0.15 +0.02,0,0],
                                            size=self.param_list['lower_arm_size'])   
            left_wrist_joint = self.add_joint( name='left_wrist',
                                                parent=left_hand,
                                                joint_type='hinge',
                                                armature=0.0028,
                                                axis=[0,1,0],
                                                pos=[0,0,0],
                                                joint_range=[-80,20],
                                                stiffness=0)                                               
            left_hand_geom_2 = self.add_geom(name='left_hand_sensor_geom',
                                            parent=left_hand,
                                            geom_type='capsule',
                                            from_point=[0.15/2-0.002 +0.02,0,0],
                                            to_point=[0.15/2+0.002 +0.02,0,0],
                                            size=self.param_list['lower_arm_size']+0.002,
                                            rgba=[1,0,0,1]) 

    def _add_actuator(self,):
        '''
        添加actuator和tendon驱动。
        由于actuator与gym的action space相关，手动添加以确保其顺序不变。
        各个关节传动比未作更改。
        '''

        tendon = self.root.child_element('tendon',{})
        left_hipknee = tendon.child_element('fixed',{'name':'left_hipknee'})
        left_hipknee.child_element('joint',{'coef':'-1','joint':'left_hip_y'})
        left_hipknee.child_element('joint',{'coef':'1','joint':'left_knee'})
        right_hipknee = tendon.child_element('fixed',{'name':'right_hipknee'})
        right_hipknee.child_element('joint',{'coef':'-1','joint':'right_hip_y'})
        right_hipknee.child_element('joint',{'coef':'1','joint':'right_knee'})

        actuator = self.root.child_element('actuator',{})
        actuator.child_element('motor',{'gear':'100','joint':'abdomen_y','name':'abdomen_y'})
        actuator.child_element('motor',{'gear':'100','joint':'abdomen_z','name':'abdomen_z'})
        actuator.child_element('motor',{'gear':'100','joint':'abdomen_x','name':'abdomen_x'})
        actuator.child_element('motor',{'gear':'100','joint':'right_hip_x','name':'right_hip_x'})
        actuator.child_element('motor',{'gear':'100','joint':'right_hip_z','name':'right_hip_z'})
        actuator.child_element('motor',{'gear':'300','joint':'right_hip_y','name':'right_hip_y'})
        actuator.child_element('motor',{'gear':'200','joint':'right_knee','name':'right_knee'})
        actuator.child_element('motor',{'gear':'50','joint':'right_ankle','name':'right_ankle'})
        actuator.child_element('motor',{'gear':'100','joint':'left_hip_x','name':'left_hip_x'})
        actuator.child_element('motor',{'gear':'100','joint':'left_hip_z','name':'left_hip_z'})
        actuator.child_element('motor',{'gear':'300','joint':'left_hip_y','name':'left_hip_y'})
        actuator.child_element('motor',{'gear':'200','joint':'left_knee','name':'left_knee'})       
        actuator.child_element('motor',{'gear':'50','joint':'left_ankle','name':'left_ankle'})
        actuator.child_element('motor',{'gear':'25' ,'joint':'right_shoulder1','name':'right_shoulder1'})
        actuator.child_element('motor',{'gear':'25' ,'joint':'right_shoulder2','name':'right_shoulder2'})
        actuator.child_element('motor',{'gear':'25' ,'joint':'right_elbow','name':'right_elbow'})
        if self.terrain_type == 'ladders':
            actuator.child_element('motor',{'gear':'25' ,'joint':'right_wrist','name':'right_wrist'})
        actuator.child_element('motor',{'gear':'25' ,'joint':'left_shoulder1','name':'left_shoulder1'})
        actuator.child_element('motor',{'gear':'25' ,'joint':'left_shoulder2','name':'left_shoulder2'})
        actuator.child_element('motor',{'gear':'25' ,'joint':'left_elbow','name':'left_elbow'})
        if self.terrain_type == 'ladders':
            actuator.child_element('motor',{'gear':'25' ,'joint':'left_wrist','name':'left_wrist'})

    def write_xml(self, file_path = 'humanoid.xml'):
        '''
        输出xml文档
        '''
        self._basic_structure() # 生成基本框架
        self.set_terrain() # 生成地形
        self._create_humanoid() # 生成机器人
        self._add_actuator() # 添加驱动
        self.generate(file_path)
        
    def reset_params(self,) -> None:
        '''
        将params list恢复至默认值
        '''
        self.param_list = {}
        self.parma_list = self.__default_param_list
        return

    def set_params(self,
                   params:Dict # humanoid robot 参数字典
                   ) -> None:
        '''
        设定params list
        '''
        self.param_list.update(params)
        if self.terrain_type == 'ladders':
            init_x = -0.3
        else:
            init_x = -1
        self.param_list['init_position'] = [init_x,0,0.76+self.param_list['shin_lenth']+self.param_list['thigh_lenth'] ] 
        return

    def set_gravity(self, gravity:float )-> None:
        # 设置环境重力。更改后需通过update_xml更新xml文档
        self.gravity = gravity
        return

    def update_xml(self, file_path='humanoid.xml'):
        '''
        更新XML文档
        为防止冲突，删除原先的root节点，重新生成XML tree。
        '''
        del self.root
        super(HumanoidXML ,self).__init__(root_tag='mujoco',gravity=self.param_list['gravity'] )
        self.write_xml(file_path=file_path)



if __name__ == "__main__":
    """     
    t = XMLTree()
    t.add_compiler()
    t.add_default()
    t.add_option()
    t.add_visual()
    t.add_asset()   
    t.generate() 
    """
    t = HumanoidXML(terrain_type='steps')
    t.write_xml(file_path="e.xml")
    #t.write_xml(file_path="gym_custom_env/assets/humanoid_exp.xml")