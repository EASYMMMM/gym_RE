'''
生成可供Mujoco使用的xml文档
'''

import xml.etree.ElementTree as ET
from typing import Any, Dict, Optional, Type, Union, List

def prettyXml(element, indent = '\t', newline = '\n', level = 0):
    '''
    从CSDN搬的美化XML文档格式的函数。
    '''
    # 判断element是否有子元素
    if element:
 
        # 如果element的text没有内容
        if element.text == None or element.text.isspace():
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
 
    # 此处两行如果把注释去掉，Element的text也会另起一行 
    #else:
        #element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
 
    temp = list(element) # 将elemnt转成list
    for subelement in temp:
        # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
        if temp.index(subelement) < (len(temp) - 1):
            subelement.tail = newline + indent * (level + 1)
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
            subelement.tail = newline + indent * level   
 
        # 对子元素进行递归操作 
        prettyXml(subelement, indent, newline, level = level + 1)

class XMLPart():
    '''
    维护一个XML元素
    '''
    def __init__(
        self,
        tag : str,
        root: bool = None,         # 是否为根节点
        parent = None ,            # 父节点
        child : Union[List,None] = None ,             # 子节点
        attributes : Union[Dict[str,str],None] = None ,    # 参数                        # 属性 
    ):
        if root:
            self.xmlpart = ET.Element(tag)
        else:
            self.xmlpart = ET.SubElement(parent.xmlpart, tag)
        # 输入参数    
        self.add_attribute(attributes)
        self.childs = dict()

    def add_attribute(self, attributes):
        '添加参数'
        for k,v in attributes.items():
            self.xmlpart.set(k,v)

    def child_element(self, tag, attributes):
        '添加子元素'
        child = XMLPart(tag = tag,
                        root = None, 
                        parent = self,
                        attributes=attributes )
        self.childs.update({tag:child})
        return child
        


class MujocoXML():
    '''
    维护一个XML树
    '''
    def __init__(
        self, 
        root_tag:str = "mujoco" ,  # 部位名称

    ):
        # 生成根节点
        self.root = XMLPart(root_tag, root = True, attributes={"model":"humanoid"} )
        self.elements : Dict[str, XMLPart] = {} # 根节点所包含的子节点的元素集，只包括往下一级的子节点，便于查找
        self.texture = list()

    def list2str(self, l):
        '''
        将输入的列表转换为带空格的字符串
        '''
        if type(l) != list:
            return str(l)
        str_list = [str(num) for num in l] 
        result_str = " ".join(str_list) 
        return result_str

    def _add_compiler(self, tag = "compiler", attributes=None) -> None:
        # 添加默认compiler参数
        compiler_attr = {"angle":"degree",
                         "inertiafromgeom":"true"}
        # 如果有自定义的参数，更新
        if attributes :
            compiler_attr.update(attributes)     
        # 添加compiler为root的子元素
        compiler = self.root.child_element(tag=tag, attributes=compiler_attr)
        self.elements.update({tag:compiler})  # 添加到元素集中

    def _add_default(self, tag = "default", attributes=None) -> None:
        default = self.root.child_element(tag=tag,attributes={})
        if attributes :
            self.elements.update({tag:default})  # 添加到元素集中
        # joint
        default_joint_attr = {"armature":"1",
                              "damping":"1",
                              "limited":"true"}
        default_joint_tag = "joint"
        defalut_joint = default.child_element(tag=default_joint_tag, 
                                              attributes=default_joint_attr)
        # geom
        default_geom_attr = {"conaffinity":"1",
                             "condim":"1",
                             "contype":"1",
                             "margin":"0.001",
                             "material":"geom",
                             "rgba":"0.8 0.6 .4 1"
                             }
        default_geom_tag = "geom"
        defalut_geom = default.child_element(tag=default_geom_tag, 
                                             attributes=default_geom_attr)
        #motor
        default_motor_attr = {"ctrllimited":"true",
                             "ctrlrange":"-.4 .4",
                             }
        default_motor_tag = "motor"
        defalut_motor = default.child_element(tag=default_motor_tag,
                                              attributes=default_motor_attr)

    def _add_option(self, tag = "option", attributes=None):
        '''
        添加option项
        '''
        option_attr = {"integrator":"RK4",
                        "iterations":"50",
                        "solver":"PGS",
                        "timestep":"0.003",
                        "gravity":"0 0 0"
                      }  
        if attributes :
            option_attr.update(attributes)      
        option = self.root.child_element(tag = tag, attributes= option_attr)
        self.elements.update({tag:option})  # 添加到元素集中

    def _add_size(self, tag = "size",attributes=None):
        '''
        添加size元素
        '''
        size_attr = {"nkey":"5",
                     "nuser_geom":"1",
                    }
        if attributes :
            size_attr.update(attributes)      
        size = self.root.child_element(tag = tag, attributes= size_attr)
        self.elements.update({tag:size})  # 添加到元素集中

    def _add_visual(self, tag="visual",attributes=None):
        '''
        添加visual元素 
        '''
        visual_attr = {}
        if attributes :    
            visual_attr.update(attributes)      
        visual = self.root.child_element(tag = tag, attributes= visual_attr)
        self.elements.update({tag:visual})  # 添加到元素集中
        visual_map_tag = "map"
        visual_map_attr = {"fogend":"5",
                           "fogstart":"3",
                            }
        visual_map = visual.child_element(tag=visual_map_tag,
                                              attributes=visual_map_attr)   

    def _add_asset(self, tag="asset",attributes=None):
        '''
        添加asset元素
        '''
        asset_attr = {}
        if attributes :
            asset_attr.update(attributes) 
        asset = self.root.child_element(tag = tag, attributes=asset_attr)
        self.elements.update({tag:asset})

        # 添加纹理
        gradient_attr = { "builtin":"gradient",
                          "height":"100",
                          "rgb1":".4 .5 .6",
                          "rgb2":"0 0 0",
                          "type":"skybox",
                          "width":"100",
                          }# gradient
        asset_texture_gradient = asset.child_element("texture",gradient_attr)
        flat_attr     = { "builtin":"flat",
                          "height":"1278",
                          "mark":"cross",
                          "markrgb":"1 1 1",
                          "name":"texgeom",
                          "random":"0.01",
                          "rgb1":"0.8 0.6 0.4",
                          "rgb2":"0.8 0.6 0.4",
                          "type":"cube",
                          "width":"127",
                          }# flat
        asset_texture_flat = asset.child_element("texture",flat_attr)
        checker_attr  = { "builtin":"checker",
                          "height":"100",
                          "name":"texplane",
                          "random":"0.01",
                          "rgb1":"0 0 0",
                          "rgb2":"0.8 0.8 0.8",
                          "type":"2d",
                          "width":"100",
                          }# flat
        asset_texture_checker = asset.child_element("texture",checker_attr)
        
        # 添加材料
        matplane_attr = { "name":"MatPlane",
                          "reflectance":"0.5",
                          "shininess":"1",
                          "specular":"1",
                          "texrepeat":"60 60",
                          "texture":"texplane",
                          }# flat
        asset_material_matplane = asset.child_element("material",matplane_attr)
        geom_attr     = { "name":"geom",
                          "texture":"texgeom",
                          "texuniform":"true",
                          }# flat
        asset_material_geom = asset.child_element("material",geom_attr)

    def add_geom(self, 
                parent:XMLPart,
                name: str,
                geom_type: str,
                size: List[Union[float,int]],
                pos: Optional[List[Union[float,int]]] = None,
                from_point: Optional[List[Union[float,int]]] = None, 
                to_point: Optional[List[Union[float,int]]] = None,
                user: Optional[List[Union[float,int]]] = None
            ) -> XMLPart:
        '''
        创建geom元素需要的参数字典
        '''
        geom_attr = {}
        geom_attr["name"]=name
        geom_attr["type"]=geom_type
        geom_attr["size"]=self.list2str(size)
        if geom_type == "sphere":  # 如果是sphere，参数为pos
            assert pos, "sphere类geom缺少参数pos"
            geom_attr["pos"] = self.list2str(pos)
        if geom_type == "capsule": # 如果是capsule，参数为fromto
            assert from_point!=None and to_point!=None , "capsule类geom缺少参数fromto"
            geom_attr["fromto"] = self.list2str(from_point) + " " + self.list2str(to_point)
        if user:
            geom_attr["user"] = self.list2str(user)
        geom_part = parent.child_element('geom',geom_attr)
        return geom_part

    def add_joint(  self, 
                    parent: XMLPart,
                    name: str,
                    joint_type: str,
                    armature: Optional[float] = None,
                    axis: Optional[List[Union[float,int]]] = None,
                    pos: Optional[List[Union[float,int]]] = None,
                    joint_range: Optional[List[Union[float,int]]] = None, 
                    stiffness: Optional[List[Union[float,int]]] = None,
                    damping: Optional[List[Union[float,int]]] = None,
                    limited: str = None,
                    ) -> XMLPart:
        '''
        创建joint元素需要的参数字典
        '''
        joint_attr = {}
        joint_attr['name'] = name
        joint_attr['type'] = joint_type
        joint_attr['armature'] = self.list2str(armature)
        if joint_type == 'free': 
            joint_attr['limited'] = limited
        else:
            joint_attr['axis'] = self.list2str(axis)
        joint_attr['pos'] = self.list2str(pos)
        if joint_range: joint_attr['range'] = self.list2str(joint_range)
        if stiffness: joint_attr['stiffness'] = self.list2str(stiffness) 
        if damping: joint_attr['damping'] = self.list2str(damping) 
        joint_part = parent.child_element('joint',joint_attr)
        return joint_part

    def pretty_xml(self) -> None:
        prettyXml(self.root.xmlpart)
        return

    def generate(self, file_path = "example.xml") -> None:
        self.pretty_xml()
        tree = ET.ElementTree(self.root.xmlpart)
        tree.write(file_path, encoding="utf-8", xml_declaration=True)


'''
TODO: 
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
                 root_tag:str = "mujoco"):
        super(HumanoidXML ,self).__init__(root_tag=root_tag)
        self.__default_param_list = { 'init_position':[0,0,1.4],
                            'head_radius' : 0.18,          # 头部半径
                            'torso_width': 0.14,           # 躯干宽
                            'torso_height': 0.425,         # 躯干高
                            'waist_lenth':0.12,            # 腰部宽
                            'pelvis_width':0.14,           # 骨盆宽
                            'thigh_lenth':0.34,            # 大腿长
                            'thigh_size':0.06,             # 大腿粗
                            'shin_lenth':0.3,              # 小腿长
                            'shin_size':0.05,              # 小腿粗
                            'upper_arm_lenth':0.2771,      # 大臂长
                            'upper_arm_size':0.04,         # 大臂粗
                            'lower_arm_lenth':0.2944,      # 小臂长
                            'lower_arm_size':0.031,        # 小臂粗
                            }
        self.param_list = self.__default_param_list.copy()

    def _basic_structure(self):
        '''
        生成XML文档的基本框架
        '''
        self._add_compiler()
        self._add_default()
        self._add_option()
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
                    terrain_type="default",
                    name: Union[str, None] = None,
                    size: List[str] = None):
        '''
        定义地形，为worldbody的子元素
        '''
        tag = "geom"
        if terrain_type == "default":
            if not name:
                name = "flatfloor"
            if not size:
                Size = "20 20 0.125"
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

        '''
        定义楼梯/梯子
        '''
        return
             
        # Define the parameters for the box geometries
        box_size = ".03 1.2 .01"
        box_rgba = "0 .9 0 1"
        box_condim = "3"
        box_friction = "1 .1 .1"

        # Define the positions using an arithmetic sequence with a common difference of 0.2
        positions = [(round(i * 0.2, 3), 0, round(i * 0.2 * 2 / 3, 3)) for i in range(11)]

        # Create a body element for the boxes,named "ground"

        ground_attr = {"name": "ground"}
        ground = self.elements["worldbody"].child_element("body", ground_attr)
        # Create a geometry for each position
        for i, pos in enumerate(positions):
            box_attr = {
                "type": "box",
                "size": box_size,
                "pos": f"{pos[0]} {pos[1]} {pos[2]}",
                "rgba": box_rgba,
                "condim": box_condim,
                "friction": box_friction,
            }
            ground.child_element(tag, box_attr)
        '''
        定义地形end
        '''

    
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
        right_hip_y_joint = self.add_joint(name='right_hip_y',
                                           parent=right_thigh,
                                           joint_type='hinge',
                                           armature=0.01,
                                           axis=[0,1,0],
                                           damping=5,
                                           pos=[0,0,0],
                                           joint_range=[-110,20],
                                           stiffness=20)
        right_hip_z_joint = self.add_joint(name='right_hip_z',
                                           parent=right_thigh,
                                           joint_type='hinge',
                                           armature=0.01,
                                           axis=[0,0,1],
                                           damping=5,
                                           pos=[0,0,0],
                                           joint_range=[-60,35],
                                           stiffness=10)
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
        left_hip_y_joint  = self.add_joint(name='left_hip_y',
                                           parent=left_thigh,
                                           joint_type='hinge',
                                           armature=0.01,
                                           axis=[0,1,0],
                                           damping=5,
                                           pos=[0,0,0],
                                           joint_range=[-110,20],
                                           stiffness=20)
        left_hip_z_joint  = self.add_joint(name='left_hip_z',
                                           parent=left_thigh,
                                           joint_type='hinge',
                                           armature=0.01,
                                           axis=[0,0,-1],
                                           damping=5,
                                           pos=[0,0,0],
                                           joint_range=[-60,35],
                                           stiffness=10)
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
    t = HumanoidXML()
    t.write_xml(file_path="e.xml")