"""
来自newbing，给了两个xml例子（由于字数限制都被截断了）
生成一段代码

"""

# This is a python program that can generate xml documents of mujoco simulation environment, with a simple biped robot.

# Importing the xml library
import xml.etree.ElementTree as ET

def prettyXml(element, indent, newline, level = 0):
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

# Defining some parameters for the robot
torso_size = 0.2 # The size of the torso sphere
torso_mass = 10 # The mass of the torso
leg_length = 0.4 # The length of each leg segment
leg_radius = 0.05 # The radius of each leg segment
leg_mass = 2 # The mass of each leg segment
arm_length = 0.3 # The length of each arm segment
arm_radius = 0.04 # The radius of each arm segment
arm_mass = 1 # The mass of each arm segment
foot_length = 0.1
foot_radius = 0.04
foot_mass = 0.5
head_mass = 1
head_radius = 0.1

# Creating the root element <mujoco>
root = ET.Element("mujoco")
root.set("model", "biped")

# Creating the <compiler> element with some attributes
compiler = ET.SubElement(root, "compiler")
compiler.set("angle", "degree")
compiler.set("coordinate", "local")
compiler.set("inertiafromgeom", "true")

# Creating the <option> element with some attributes
option = ET.SubElement(root, "option")
option.set("integrator", "RK4")
option.set("timestep", "0.01")

# Creating the <default> element with some subelements and attributes
default = ET.SubElement(root, "default")
joint = ET.SubElement(default, "joint")
joint.set("armature", "1")
joint.set("damping", "1")
joint.set("limited", "true")
geom = ET.SubElement(default, "geom")
geom.set("conaffinity", "0")
geom.set("condim", "3")
geom.set("density", "5.0")
geom.set("friction", "1 0.5 0.5")
geom.set("margin", "0.01")
geom.set("rgba", "0.8 0.6 0.4 1")

# Creating the <asset> element with some subelements and attributes
asset = ET.SubElement(root, "asset")

texture_skybox = ET.SubElement(asset, "texture") # A texture for the skybox
texture_skybox.set("builtin","gradient") 
texture_skybox.set("height","100") 
texture_skybox.set("rgb1","1 1 1") 
texture_skybox.set("rgb2","0 0 0") 
texture_skybox.set("type","skybox") 
texture_skybox.set("width","100")

texture_geom = ET.SubElement(asset, "texture") # A texture for the robot geom
texture_geom .set ("builtin","flat") 
texture_geom .set ("height","1278") 
texture_geom .set ("mark","cross") 
texture_geom .set ("markrgb","1 1 1") 
texture_geom .set ("name","texgeom") 
texture_geom .set ("random","0.01") 
texture_geom .set ("rgb1","0.8 0.6 0.4") 
texture_geom .set ("rgb2","0.8 0.6 0.4") 
texture_geom .set ("type","cube")

texture_plane = ET.SubElement(asset, "texture") # A texture for the ground plane
texture_plane.set("builtin","checker") 
texture_plane.set("height","100") 
texture_plane.set("name","texplane") 
texture_plane.set("rgb1","0 0 0") 
texture_plane.set("rgb2","0.8 0.8 0.8") 
texture_plane.set("type","2d") 
texture_plane.set("width","100")

material_geom = ET.SubElement(asset, "material") # A material for the robot geom
material_geom .set ("name","geom")
material_geom .set ("texture","texgeom")
material_geom .set ("texuniform","true")

material_plane = ET.SubElement(asset, "material") # A material for the ground plane
material_plane .set ("name","MatPlane")
material_plane .set ("reflectance","0.5")
material_plane .set ("shininess","1")
material_plane .set ("specular","1")
material_plane .set ("texrepeat","60 60")
material_plane .set ("texture","texplane")

# Creating the <worldbody> element with some subelements and attributes
worldbody = ET.SubElement(root, "worldbody")

light = ET.SubElement(worldbody, "light") # A light source
light.set("cutoff", "100")
light.set("diffuse", "1 1 1")
light.set("dir", "-0 0 -1.3")
light.set("directional", "true")
light.set("exponent", "1")
light.set("pos", "0 0 1.3")
light.set("specular", ".1 .1 .1")

floor = ET.SubElement(worldbody, "geom") # A ground plane
floor.set("conaffinity", "1")
floor.set("condim", "3")
floor.set("material", "MatPlane")
floor.set("name", "floor")
floor.set("pos", "0 0 0")
floor.set("rgba", "0.8 0.9 0.8 1")
floor.set("size", "40 40 40")
floor.set("type", "plane")

torso = ET.SubElement(worldbody, "body") # The torso of the robot
torso_geom = ET.SubElement(torso, "geom") # The torso geom
torso_geom .set ("name" ,   "torso_geom" )
torso_geom .set ("pos" ,   f" { torso_size }   { torso_size }   { torso_size } ")
torso_geom .set ("size" ,   f" { torso_size } ")
torso_geom .set ("type" ,   "sphere" )
root_joint = ET.SubElement(torso, "joint" ) # The root joint of the robot
root_joint . set ( "armature" ,   f" { torso_mass } ")
root_joint . set ( "damping" ,   f" { torso_mass /10} ")
root_joint . set ( "name" ,   "root" )
root_joint . set ( "pos" ,   f" { torso_size }   { torso_size }   { torso_size } ")
root_joint . set ( "type" ,   "free" )
torso_inertial = ET.SubElement(torso, "inertial") # The torso inertial
torso_inertial . set ( "mass" ,   f" { torso_mass } ")
torso_inertial . set ( "pos" ,   f" { torso_size }   { torso_size }   { torso_size } ")

right_thigh = ET.SubElement(torso, "body") # The right thigh of the robot
right_thigh_geom = ET.SubElement(right_thigh, "geom") # The right thigh geom
right_thigh_geom . set ( "fromto" ,   f"0 0 0 0 -{ leg_length /2} -{ leg_length /2}")
right_thigh_geom . set ( "name" ,   "right_thigh" )
right_thigh_geom . set ( "size" ,   f" { leg_radius } ")
right_thigh_geom . set ( "type" ,   "capsule" )
right_hip = ET.SubElement(right_thigh, "joint") # The right hip joint
right_hip . set ( "axis" ,   "0 1 1" )
right_hip . set ( "name" ,   "right_hip" )
right_hip . set ( "pos" ,   f"0 -{ leg_length /2} -{ leg_length /2}")
right_hip . set ( "range" ,"-30 30" )
right_hip . set ( "type" , "hinge" )
right_thigh_inertial = ET.SubElement(right_thigh, "inertial") # The right thigh inertial
right_thigh_inertial.set("mass", f"{leg_mass}")
right_thigh_inertial.set("pos", f"0 -{leg_length/2} -{leg_length/2}")

left_thigh = ET.SubElement(torso, "body") # The left thigh of the robot
left_thigh_geom = ET.SubElement(left_thigh, "geom") # The left thigh geom
left_thigh_geom.set("fromto", f"0 0 0 0 {leg_length/2} -{leg_length/2}")
left_thigh_geom.set("name", "leftthight")
left_thigh_geom.set("size", f"{leg_radius}")
left_thigh_geom.set("type", "capsule")
left_hip = ET.SubElement(left_thigh, "joint") # The left hip joint
left_hip.set("axis", "0 -1 1")
left_hip.set("name", "lefthip")
left_hip.set("pos", f"0 {leg_length/2} -{leg_length/2}")
left_hip.set("range", "-30 30")
left_hip.set("type", "hinge")
left_thigh_inertial = ET.SubElement(left_thigh, "inertial") # The left thigh inertial
left_thigh_inertial.set("mass", f"{leg_mass}")
left_thigh_inertial.set("pos", f"0 {leg_length/2} -{leg_length/2}")

right_shin = ET.SubElement(right_thigh, "body") # The right shin of the robot
right_shin_geom = ET.SubElement(right_shin, "geom") # The right shin geom
right_shin_geom.set("fromto", f"0 0 0 0 -{leg_length/2} -{leg_length/2}")
right_shin_geom.set("name", "rightshin")
right_shin_geom.set("size", f"{leg_radius}")
right_shin_geom.set("type", "capsule")
right_knee = ET.SubElement(right_shin, "joint") # The right knee joint
right_knee.set("axis", "1 1 0")
right_knee.set("name", "rightknee")
right_knee.set("pos", f"0 -{leg_length/2} -{leg_length/2}")
right_knee.set("range", "-150 0")
right_knee.set("type", "hinge")
right_shin_inertial = ET.SubElement(right_shin, "inertial") # The right shin inertial
right_shin_inertial . set ( "mass" ,   f" { leg_mass } ")
right_shin_inertial.set("pos", f"0 -{leg_length/2} -{leg_length/2}")

left_shin = ET.SubElement(left_thigh, "body") # The left shin of the robot
left_shin_geom = ET.SubElement(left_shin, "geom") # The left shin geom
left_shin_geom.set("fromto", f"0 0 0 0 {leg_length/2} -{leg_length/2}")
left_shin_geom.set("name", "leftshin")
left_shin_geom.set("size", f"{leg_radius}")
left_shin_geom.set("type", "capsule")
left_knee = ET.SubElement(left_shin, "joint") # The left knee joint
left_knee.set("axis", "-1 1 0")
left_knee.set("name", "leftknee")
left_knee.set("pos", f"0 {leg_length/2} -{leg_length/2}")
left_knee.set("range", "-150 0")
left_knee.set("type", "hinge")
left_shin_inertial = ET.SubElement(left_shin, "inertial") # The left shin inertial
left_shin_inertial . set ( "mass" ,   f" { leg_mass } ")
left_shin_inertial.set("pos", f"0 {leg_length/2} -{leg_length/2}")

right_foot = ET.SubElement(right_shin, "body") # The right foot of the robot
right_foot_geom = ET.SubElement(right_foot, "geom") # The right foot geom
right_foot_geom.set("fromto", f"0 0 0 {foot_length} 0 0")
right_foot_geom.set("name", "rightfoot")
right_foot_geom.set("size", f"{foot_radius}")
right_foot_geom.set("type", "capsule")
right_ankle = ET.SubElement(right_foot, "joint") # The right ankle joint
right_ankle.set("axis", "-1 1 1")
right_ankle.set("name", "rightankle")
right_ankle.set("pos", f"0 -{leg_length/2} -{leg_length/2}")
right_ankle.set("range", "-45 45")
right_ankle.set("type", "hinge")
right_foot_inertial = ET.SubElement(right_foot, "inertial") # The right foot inertial
right_foot_inertial . set ( "mass" ,   f" { foot_mass } ")
right_foot_inertial.set("pos", f"{foot_length/2} 0 0")

left_foot = ET.SubElement(left_shin, "body") # The left foot of the robot
left_foot_geom = ET.SubElement(left_foot, "geom") # The left foot geom
left_foot_geom.set("fromto", f"0 0 0 -{foot_length} 0 0")
left_foot_geom.set("name", "leftfoot")
left_foot_geom.set("size", f"{foot_radius}")
left_foot_geom.set("type", "capsule")
left_ankle = ET.SubElement(left_foot, "joint") # The left ankle joint
left_ankle.set("axis", "1 1 -1")
left_ankle.set("name", "leftankle")
left_ankle.set("pos", f"0 {leg_length/2} -{leg_length/2}")
left_ankle.set("range", "-45 45")
left_ankle.set("type", "hinge")
left_foot_inertial = ET.SubElement(left_foot, "inertial") # The left foot inertial
left_foot_inertial . set ( "mass" ,   f" { foot_mass } ")
left_foot_inertial.set("pos", f"-{foot_length/2} 0 0")

head = ET.SubElement(torso, "body") # The head of the robot
head_geom = ET.SubElement(head, "geom") # The head geom
head_geom.set("name", "head")
head_geom.set("pos", f"0 {torso_size} {torso_size}")
head_geom.set("size", f"{head_radius}")
head_geom.set("type", "sphere")
neck = ET.SubElement(head, "joint") # The neck joint
neck.set("axis", "1 1 1")
neck.set("name", "neck")
neck.set("pos", f"0 {torso_size} {torso_size}")
neck.set("range", "-90 90")
neck.set("type", "hinge")
head_inertial = ET.SubElement(head, "inertial") # The head inertial
head_inertial . set ( "mass" ,   f" { head_mass } ")
head_inertial.set("pos", f"0 {torso_size} {torso_size}")

right_arm = ET.SubElement(torso, "body") # The right arm of the robot
right_arm_geom = ET.SubElement(right_arm, "geom") # The right arm geom
right_arm_geom.set("fromto", f"0 0 0 -{arm_length/2} 0 0")
right_arm_geom.set("name", "rightarm")
right_arm_geom.set("size", f"{arm_radius}")
right_arm_geom.set("type", "capsule")
right_shoulder = ET.SubElement(right_arm, "joint") # The right shoulder joint
right_shoulder.set("axis", "-1 -1 -1")
right_shoulder.set("name", "rightshoulder")
right_shoulder.set("pos", f"0 {torso_size} {torso_size}")
right_shoulder.set("range", "-180 180")
right_shoulder.set("type", "hinge")
right_arm_inertial = ET.SubElement(right_arm, "inertial") # The right arm inertial
right_arm_inertial . set ( "mass" ,   f" { arm_mass } ") 
right_arm_inertial.set("pos", f"-{arm_length/4} 0 0")

left_arm = ET.SubElement(torso, "body") # The left arm of the robot
left_arm_geom = ET.SubElement(left_arm, "geom") # The left arm geom
left_arm_geom.set("fromto", f"0 0 0 {arm_length/2} 0 0")
left_arm_geom.set("name", "leftarm")
left_arm_geom.set("size", f"{arm_radius}")
left_arm_geom.set("type", "capsule")
left_shoulder = ET.SubElement(left_arm, "joint") # The left shoulder joint
left_shoulder.set("axis", "1 -1 -1")
left_shoulder.set("name", "leftshoulder")
left_shoulder.set("pos", f"0 {torso_size} {torso_size}")
left_shoulder.set("range", "-180 180")
left_shoulder.set("type", "hinge")
left_arm_inertial = ET.SubElement(left_arm, "inertial") # The left arm inertial
left_arm_inertial . set ("mass" ,   f" { arm_mass } ")
right_arm_inertial.set("pos", f"{arm_length/4} 0 0")

prettyXml(root, '\t', '\n')   
tree = ET.ElementTree(root)
tree.write("example.xml", encoding="utf-8", xml_declaration=True)