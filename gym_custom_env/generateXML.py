import xml.etree.ElementTree as ET
from typing import Any, Dict, Optional, Type, Union, List
import sys,os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from xml_util import XMLPart,MujocoXML ,prettyXml


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
                 ):
        super(HumanoidXML ,self).__init__(root_tag=root_tag)

        self.terrain_type = terrain_type
        terrain_type_list = ('default','steps','ladders') # 默认平地，台阶，梯子
        assert self.terrain_type in terrain_type_list, 'ERROR:Undefined terrain type'
        init_pos = {'default':[-1,0,1.4] , 'steps':[-1,0,1.4], 'ladders':[-0.3,0,1.4]}[self.terrain_type]
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
                            'lower_arm_size':0.031,        # 小臂粗 0.2944
                            }
        self.param_list = self.__default_param_list.copy()
        self.geom_name_list = list() 

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
            self.geom_name_list.append(name)

        # 定义梯子
        if terrain_type == "ladders":
            # Define the parameters for the box geometries
            box_size = ".06 1.2 .01"
            box_rgba = "0 .9 0 1"
            box_condim = "3"
            box_friction = "1 .1 .1"

            # Define the positions using an arithmetic sequence with a common difference of 0.2
            positions = [(round(i * 0.1, 3), 0, round(i * 0.2 * 3 / 2, 3)) for i in range(11)]

            # Create a geometry for each position
            for i, pos in enumerate(positions):
                box_attr = {
                    "name": f"ladder{i + 1}",
                    "type": "box",
                    "size": box_size,
                    "pos": f"{pos[0]} {pos[1]} {pos[2]}",
                    "rgba": box_rgba,
                    "condim": box_condim,
                    "friction": box_friction,
                }
                self.elements["worldbody"].child_element(tag, box_attr)
        
        # 定义楼梯
        if terrain_type == 'steps':     
            # Define the parameters for the box geometries
            box_size = "0.3 1.2 0.1"
            # should be equal to the xyz in box_size
            box_x = 0.3
            box_y = 1.2
            box_z = 0.1

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
        camera = torso.child_element("camera",camera_attr)
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
        self.geom_name_list.append('torso_geom')
        
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
        foot_pos = self.param_list['shin_lenth']+self.param_list['shin_size']+0.1
        right_foot_attr = {'name':'right_foot', 'pos':f'0 0 -{foot_pos}'}  
        right_foot = right_shin.child_element('body',right_foot_attr)
        right_foot_geom = self.add_geom(name='right_foot_geom',
                                        parent=right_foot,
                                        geom_type='sphere',
                                        pos=[0,0,0.1],
                                        size=0.075,
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
        left_foot = left_shin.child_element('body',left_foot_attr)
        left_foot_geom  = self.add_geom(name='left_foot_geom',
                                        parent=left_foot,
                                        geom_type='sphere',
                                        pos=[0,0,0.1],
                                        size=0.075,
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
                                               axis=[2,1,1],
                                               pos=[0,0,0],
                                               joint_range=[-85,60],
                                               stiffness=1)
        right_shoulder2_joint = self.add_joint(name='right_shoulder2',
                                               parent=right_upper_arm,
                                               joint_type='hinge',
                                               armature=0.0051,
                                               axis=[0,-1,1],
                                               pos=[0,0,0],
                                               joint_range=[-85,60],
                                               stiffness=1)
        upper_arm_pos = pow(self.param_list['upper_arm_lenth']*self.param_list['upper_arm_lenth']/3,0.5)                                       
        right_upper_arm_geom = self.add_geom(name='right_upper_arm_geom',
                                             parent=right_upper_arm,
                                             geom_type='capsule',
                                             from_point=[0,0,0],
                                             to_point=[upper_arm_pos,-upper_arm_pos,-upper_arm_pos],
                                             size=self.param_list['upper_arm_size'])
        lower_arm_begin_pos = upper_arm_pos+self.param_list["upper_arm_size"]/2
        right_lower_arm_attr = {'name':'right_lower_arm', 'pos':f'{lower_arm_begin_pos} -{lower_arm_begin_pos} -{lower_arm_begin_pos}' }   
        right_lower_arm = right_upper_arm.child_element('body',right_lower_arm_attr)
        right_elbow_joint = self.add_joint( name='right_elbow',
                                            parent=right_lower_arm,
                                            joint_type='hinge',
                                            armature=0.0028,
                                            axis=[0,-1,1],
                                            pos=[0,0,0],
                                            joint_range=[-90,50],
                                            stiffness=0)
        lower_arm_pos = pow(self.param_list['lower_arm_lenth']*self.param_list['lower_arm_lenth']/3,0.5) 
        right_lower_arm_geom = self.add_geom(name='right_lower_arm_geom',
                                             parent=right_lower_arm,
                                             geom_type='capsule',
                                             from_point=[0.01,0.01,0.01],
                                             to_point=[lower_arm_pos,lower_arm_pos,lower_arm_pos],
                                             size=self.param_list['lower_arm_size'])
        right_hand_geom = self.add_geom(name='right_hand',
                                        parent=right_lower_arm,
                                        geom_type='sphere',
                                        pos=[lower_arm_pos+0.01,lower_arm_pos+0.01,lower_arm_pos+0.01],
                                        size=self.param_list['lower_arm_size']*1.2)       
        # 左臂
        left_upper_arm_attr = { 'name':'left_upper_arm','pos':f'0 {shoulder_pos} 0.06'}
        left_upper_arm = torso.child_element('body',left_upper_arm_attr)
        left_shoulder1_joint = self.add_joint(name='left_shoulder1',
                                               parent=left_upper_arm,
                                               joint_type='hinge',
                                               armature=0.0068,
                                               axis=[2,-1,1],
                                               pos=[0,0,0],
                                               joint_range=[-60,85],
                                               stiffness=1)
        left_shoulder2_joint = self.add_joint(name='left_shoulder2',
                                               parent=left_upper_arm,
                                               joint_type='hinge',
                                               armature=0.0051,
                                               axis=[0,1,1],
                                               pos=[0,0,0],
                                               joint_range=[-60,85],
                                               stiffness=1)
        left_upper_arm_geom = self.add_geom(name='left_upper_arm_geom',
                                             parent=left_upper_arm,
                                             geom_type='capsule',
                                             from_point=[0,0,0],
                                             to_point=[upper_arm_pos,upper_arm_pos,-upper_arm_pos],
                                             size=self.param_list['upper_arm_size'])
        left_lower_arm_attr = {'name':'left_lower_arm', 'pos':f'{lower_arm_begin_pos} {lower_arm_begin_pos} -{lower_arm_begin_pos}' }   
        left_lower_arm = left_upper_arm.child_element('body',left_lower_arm_attr)
        left_elbow_joint = self.add_joint( name='left_elbow',
                                            parent=left_lower_arm,
                                            joint_type='hinge',
                                            armature=0.0028,
                                            axis=[0,-1,-1],
                                            pos=[0,0,0],
                                            joint_range=[-90,50],
                                            stiffness=0)
        left_lower_arm_geom = self.add_geom(name='left_lower_arm_geom',
                                             parent=left_lower_arm,
                                             geom_type='capsule',
                                             from_point=[0.01,-0.01,0.01],
                                             to_point=[lower_arm_pos,-lower_arm_pos,lower_arm_pos],
                                             size=self.param_list['lower_arm_size'])
        left_hand_geom = self.add_geom(name='left_hand',
                                        parent=left_lower_arm,
                                        geom_type='sphere',
                                        pos=[lower_arm_pos+0.01,-(lower_arm_pos+0.01),lower_arm_pos+0.01],
                                        size=self.param_list['lower_arm_size']*1.2)                                    
    
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
        actuator.child_element('motor',{'gear':'100','joint':'left_hip_x','name':'left_hip_x'})
        actuator.child_element('motor',{'gear':'100','joint':'left_hip_z','name':'left_hip_z'})
        actuator.child_element('motor',{'gear':'300','joint':'left_hip_y','name':'left_hip_y'})
        actuator.child_element('motor',{'gear':'200','joint':'left_knee','name':'left_knee'})       
        actuator.child_element('motor',{'gear':'25' ,'joint':'right_shoulder1','name':'right_shoulder1'})
        actuator.child_element('motor',{'gear':'25' ,'joint':'right_shoulder2','name':'right_shoulder2'})
        actuator.child_element('motor',{'gear':'25' ,'joint':'right_elbow','name':'right_elbow'})
        actuator.child_element('motor',{'gear':'25' ,'joint':'left_shoulder1','name':'left_shoulder1'})
        actuator.child_element('motor',{'gear':'25' ,'joint':'left_shoulder2','name':'left_shoulder2'})
        actuator.child_element('motor',{'gear':'25' ,'joint':'left_elbow','name':'left_elbow'})

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
        return

    def update_xml(self, file_path='humanoid.xml'):
        '''
        更新XML文档
        '''
        del self.root
        super(HumanoidXML ,self).__init__(root_tag='mujoco')
        self.write_xml(file_path=file_path)

    def get_geom_namelist(self):
        '''
        获得XML的全部几何体名称
        TODO: 在每个定义geom后，添加self.geom_name_list.append(xxx)
        '''
        return self.geom_name_list



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
    t = HumanoidXML(terrain_type='ladders')
    t.write_xml(file_path="e.xml")
    #t.write_xml(file_path="gym_custom_env/assets/humanoid_exp.xml")