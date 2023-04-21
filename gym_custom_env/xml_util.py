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
        attributes : Union[Dict[str,str],None] = None ,    # 参数
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
    Mujoco仿真环境XML树 基础模板
    '''
    def __init__(
        self, 
        root_tag:str = "mujoco" ,  # 部位名称
        gravity:float = -9.81
    ):
        # 生成根节点
        self.gravity = gravity
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
                        "gravity":f"0 0 {self.gravity}"
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
                user: Optional[List[Union[float,int]]] = None,
                rgba: Optional[List[Union[float,int]]] = None,
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
        if rgba:
            geom_attr["rgba"] = self.list2str(rgba)
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
        if stiffness != None : joint_attr['stiffness'] = self.list2str(stiffness) 
        if damping != None : joint_attr['damping'] = self.list2str(damping) 
        joint_part = parent.child_element('joint',joint_attr)
        return joint_part

    def pretty_xml(self) -> None:
        prettyXml(self.root.xmlpart)
        return

    def generate(self, file_path = "example.xml") -> None:
        self.pretty_xml()
        tree = ET.ElementTree(self.root.xmlpart)
        tree.write(file_path, encoding="utf-8", xml_declaration=True)