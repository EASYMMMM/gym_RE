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
        self.humanoid_init_pos = "0 0 1.4" # humanoid出生位置

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
    def __init__(self, 
                 root_tag:str = "mujoco"):
        super(HumanoidXML ,self).__init__(root_tag=root_tag)
        self.param_list = { 'head_radius' : 0.18,
                            'torso_lenth': 0.34,
        }

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
                    name: Union[str, None] = None , 
                    size: List[str] = None):
        '''
        定义地形，为worldbody的子元素
        '''
        tag = "geom"
        if terrain_type=="default":
            if not name:
                name = "flatfloor" 
            if not size:
                Size = "20 20 0.125"
            else:
                Size = self.list2str(size)
            terrain_attr = {"condim":"3",
                            "friction":"1 .1 .1",
                            "material":"MatPlane",
                            "name":name,
                            "pos":"0 0 0",
                            "rgba":"0.8 0.9 0.8 1",
                            "size":Size,
                            "type":"plane",}
        self.elements["worldbody"].child_element(tag, terrain_attr) 
    
    def _create_torso(self ) -> XMLPart:
        '''
        创建躯干
        '''
        worldbody = self.elements["worldbody"]
        torso_attr = {"name":"torso", "pos":self.humanoid_init_pos}
        torso = worldbody.child_element("body", torso_attr)  # 躯干
        camera_attr = {'name':'track',
                       'mode':'trackcom',
                       'pos':'0 -4 0',
                       'xyaxes':'1 0 0 0 0 1',}
        camera = torso.child_element("camera",camera_attr)
        root_joint = self.add_joint(name='root',
                                     parent = torso,
                                     joint_type='free', 
                                     armature = 0 , 
                                     damping = 0, 
                                     limited = 'false',
                                     pos = [0,0,0],
                                     stiffness = 0 )
        
        torso_geom = self.add_geom(name = 'torso_geom',
                                   parent = torso,
                                   geom_type = 'capsule',
                                   from_point = [0, -self.param_list['torso_lenth']/2, 0],
                                   to_point = [0, self.param_list['torso_lenth']/2, 0],
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
                                    from_point= [-0.01, -0.06, -0.12],
                                    to_point= [-0.01, 0.06, -0.12])
        return torso

    def create_humanoid(self,):
        '''
        创建humanoid机器人
        '''
        torso = self._create_torso()  
        
    def write_xml(self, file_path = 'humanoid.xml'):
        '''
        输出xml文档
        '''
        self._basic_structure() # 生成基本框架
        self.set_terrain() # 生成地形
        self.create_humanoid() # 生成机器人
        self.generate(file_path)

    def set_params(self, **params):
        pass

    def update(self, **params):
        pass


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