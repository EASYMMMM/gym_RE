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
        


class XMLTree():
    '''
    维护一个XML树
    '''
    def __init__(
        self, 
        root_tag:str = "mujoco" ,  # 部位名称

    ):
        # 生成根节点
        self.root = XMLPart(root_tag, root = True, attributes={"mudel":"humanoid"} )
        self.elements = dict() # 根节点所包含的子节点的元素集，只包括往下一级的子节点，便于查找
        self.texture = list()

    def add_compiler(self, tag = "compiler", attributes=None) -> None:
        # 添加默认compiler参数
        compiler_attr = {"angle":"degree",
                         "inertiafromgeom":"true"}
        # 如果有自定义的参数，更新
        if attributes :
            compiler_attr.update(attributes)     
        # 添加compiler为root的子元素
        compiler = self.root.child_element(tag=tag, attributes=compiler_attr)
        self.elements.update({tag:compiler})  # 添加到元素集中

    def add_default(self, tag = "default", attributes=None) -> None:
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
        default_geom_tag = "joint"
        defalut_geom = default.child_element(tag=default_geom_tag, 
                                             attributes=default_geom_attr)
        #motor
        default_motor_attr = {"ctrllimited":"true",
                             "ctrlrange":"-.4 .4",
                             }
        default_motor_tag = "joint"
        defalut_motor = default.child_element(tag=default_motor_tag,
                                              attributes=default_motor_attr)


    def add_option(self, tag = "option", attributes=None):
        '''
        添加option项
        '''
        option_attr = {"integrator":"RK4",
                        "iterations":"50",
                        "solver":"PGS",
                        "timestep":"0.003",
                      }  
        if attributes :
            option_attr.update(attributes)      
        option = self.root.child_element(tag = tag, attributes= option_attr)
        self.elements.update({tag:option})  # 添加到元素集中

    def add_size(self, tag = "size",attributes=None):
        '''
        TODO: 添加size元素   <size nkey="5" nuser_geom="1"/>
        '''
        size_attr = {"nkey":"5",
                     "nuser_geom":"1",
                    }
        if attributes :
            size_attr.update(attributes)      
        size = self.root.child_element(tag = tag, attributes= size_attr)
        self.elements.update({tag:size})  # 添加到元素集中


    def add_visual(self, tag="visual",attributes=None):
        '''
        TODO: 添加visual元素 
        <visual>
            <map fogend="5" fogstart="3"/>
        </visual>
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

    def add_asset(self, tag="asset",attributes=None):
        '''
        TODO: 添加asset元素
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
                          "specular":"1",
                          "texrepeat":"60 60",
                          "texture":"texplane",
                          }# flat
        asset_material_geom = asset.child_element("material",geom_attr)

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
class HumanoidXML(XMLTree):
    def __init__(self, 
                 root_tag:str = "mujoco"):
        super(HumanoidXML,self).__init__(root_tag=root_tag)
        
    def base_structure(self,):
        '''
        生成XML文档的基本框架
        '''
        self.add_compiler()
        self.add_default()
        self.add_option()
        self.add_visual()
        self.add_asset()

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
    t.base_structure()
    t.generate(file_path="e.xml")