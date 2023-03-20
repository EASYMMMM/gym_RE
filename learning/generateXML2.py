import xml.etree.ElementTree as ET

# 创建xml文档根节点
root = ET.Element("mujoco")
root.set("model", "simple_robot")

# 添加物理引擎和默认设置
option = ET.SubElement(root, "option")
ET.SubElement(option, "timestep").text = "0.01"
ET.SubElement(option, "solver_type").text = "PGS"
ET.SubElement(option, "gravity").text = "0 0 -9.81"

# 添加worldbody元素
worldbody = ET.SubElement(root, "worldbody")

# 添加地面元素
ground = ET.SubElement(worldbody, "geom")
ground.set("type", "plane")
ground.set("size", "500 500 0.1")
ground.set("pos", "0 0 0")

# 添加机器人元素
robot = ET.SubElement(worldbody, "body")
robot.set("name", "robot")
robot.set("pos", "0 0 1.1")

# 添加躯干元素
torso = ET.SubElement(robot, "body")
torso.set("name", "torso")
torso.set("pos", "0 0 0.5")
torso.set("conaffinity", "1")
torso.set("contype", "1")
torso_geom = ET.SubElement(torso, "geom")
torso_geom.set("name", "torso_geom")
torso_geom.set("type", "capsule")
torso_geom.set("fromto", "0 0 0 0 0 -0.3")
torso_geom.set("size", "0.15")
torso_joint = ET.SubElement(torso, "joint")
torso_joint.set("name", "torso_joint")
torso_joint.set("type", "free")

# 添加头部元素
head = ET.SubElement(torso, "body")
head.set("name", "head")
head.set("pos", "0 0 0.3")
head.set("conaffinity", "1")
head.set("contype", "1")
head_geom = ET.SubElement(head, "geom")
head_geom.set("name", "head_geom")
head_geom.set("type", "sphere")
head_geom.set("size", "0.1")
head_joint = ET.SubElement(head, "joint")
head_joint.set("name", "head_joint")
head_joint.set("type", "hinge")
head_joint.set("axis", "1 0 0")
head_joint.set("pos", "0 0 0")

# 添加左臂元素
left_arm = ET.SubElement(torso, "body")
left_arm.set("name", "left_arm")
left_arm.set("pos", "0.15 0 0")
left_arm.set("conaffinity", "1")
left_arm.set("contype", "1")
left_arm_geom = ET.SubElement(left_arm, "geom")
left_arm_geom.set("name", "left_arm_geom")
left_arm_geom.set("type", "capsule")
left_arm_geom.set
