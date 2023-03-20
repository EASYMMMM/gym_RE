'''
ask : 能否写一段python程序，使其具有以下功能： 1. 生成一个可用于mujoco仿真环境的xml文档，
具有一个简单的仿人机器人。 2. 仿人机器人各个关节的长度参数可以调整，如手臂长度、躯干长度等。

chatGPT:
'''
# python learning/generateXML.py 

import xml.etree.ElementTree as ET

# 创建根元素
root = ET.Element("mujoco")

# 添加 mjmodel 元素
mjmodel = ET.SubElement(root, "mjmodel", time="0", timestep="0.01", gravity="0 0 -9.81", 
                        invmartrix_solimp="0.9 0.95 0.001", rabbit_viscous="0", solver_error_tolerance="0.0001", 
                        solver_residue_tolerance="0.0001", solver_velocity_tolerance="1e-12")

# 添加 option 元素
option = ET.SubElement(mjmodel, "option", integrator="RK4", timestep="0.001")

# 添加 size 元素
size = ET.SubElement(mjmodel, "size", nxmax="500", nymax="500", nconmax="100")

# 添加 body 元素
body = ET.SubElement(mjmodel, "body", name="torso", pos="0 0 1.2")
geom = ET.SubElement(body, "geom", type="capsule", fromto="0 0 0 0 0 -0.3", size="0.3")

# 添加 head 元素
head = ET.SubElement(body, "body", name="head", pos="0 0 0.3")
geom = ET.SubElement(head, "geom", type="sphere", size="0.2")

# 添加 left arm 元素
leftarm = ET.SubElement(body, "body", name="leftarm", pos="-0.3 0 0")
joint = ET.SubElement(leftarm, "joint", name="leftshoulder", type="hinge", limited="true", damping="1.0")
geom = ET.SubElement(leftarm, "geom", type="capsule", fromto="0 0 0 0 -0.3 0", size="0.1")
hand = ET.SubElement(leftarm, "body", name="lefthand", pos="0 -0.3 0")
geom = ET.SubElement(hand, "geom", type="sphere", size="0.15")

# 添加 right arm 元素
rightarm = ET.SubElement(body, "body", name="rightarm", pos="0.3 0 0")
joint = ET.SubElement(rightarm, "joint", name="rightshoulder", type="hinge", limited="true", damping="1.0")
geom = ET.SubElement(rightarm, "geom", type="capsule", fromto="0 0 0 0 -0.3 0", size="0.1")
hand = ET.SubElement(rightarm, "body", name="righthand", pos="0 -0.3 0")
geom = ET.SubElement(hand, "geom", type="sphere", size="0.15")

# 添加 left leg 元素
leftleg = ET.SubElement(body, "body", name="leftleg", pos="-0.15 0 0", euler="0 -0.785 0")
joint = ET.SubElement(leftleg, "joint", name="lefthip", type="hinge", limited="true", damping="1.0")
geom = ET.SubElement(leftleg, "geom", type="capsule", fromto="0 0 0 0 -0.4 0", size="0.15")
foot = ET.SubElement(leftleg, "body", name="leftfoot", pos="0 -0.4 -0.1", euler="0 -0.785 0")
geom = ET.SubElement(foot, "geom", type="box", size="0.2 0.5 0.1")

# 添加 right leg 元素
rightleg = ET.SubElement(body, "body", name="rightleg", pos="0.15 0 0", euler="0 0.785 0")
joint = ET.SubElement(rightleg, "joint", name="righthip", type="hinge", limited="true", damping="1.0")
geom = ET.SubElement(rightleg, "geom", type="capsule", fromto="0 0 0 0 -0.4 0", size="0.15")
foot = ET.SubElement(rightleg, "body", name="rightfoot", pos="0 -0.4 -0.1", euler="0 0.785 0")
geom = ET.SubElement(foot, "geom", type="box", size="0.2 0.5 0.1")

# 将元素写入文件
tree = ET.ElementTree(root)
tree.write("example.xml", encoding="utf-8", xml_declaration=True)
