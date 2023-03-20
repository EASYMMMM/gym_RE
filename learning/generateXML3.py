'''
来自new Bing

Can you write a code in python, which has the functions below: 
1. Generate a .xml file used by Mujoco simulate enviornment, contains a simple humanoid robot. 
2. Has a ground in it. 3. The parameters of the robot, such as the length of right arm, can be adjusted.


python learning/generateXML3.py
'''


# Importing minidom for xml manipulation
from xml.dom import minidom

# Creating a document object
xdocument = minidom.Document()

# Creating a root element with mujoco tag
root = xdocument.createElement("mujoco")
root.setAttribute("model", "humanoid")

# Creating a worldbody element with ground and body tags
worldbody = xdocument.createElement("worldbody")
ground = xdocument.createElement("geom")
ground.setAttribute("type", "plane")
ground.setAttribute("size", "100 100 0.1")
body = xdocument.createElement("body")
body.setAttribute("name", "humanoid")
body.setAttribute("pos", "0 0 1.4")

# Creating a function to generate joints and geoms for each body part
def create_body_part(name, type, size, axis="z", pos="0 0 0"):
    joint = xdocument.createElement("joint")
    joint.setAttribute("name", name + "_joint")
    joint.setAttribute("type", type)
    joint.setAttribute("axis", axis)
    joint.setAttribute("pos", pos)
    geom = xdocument.createElement("geom")
    geom.setAttribute("name", name + "_geom")
    geom.setAttribute("type", "capsule" if type == "hinge" else "sphere")
    geom.setAttribute("size", size)
    return joint, geom

# Creating body parts with adjustable parameters
head_joint, head_geom = create_body_part(name="head",
                                         type="free",
                                         size="0.15",
                                         pos="0 0 1.6")

torso_joint, torso_geom = create_body_part(name="torso",
                                           type="free",
                                           size="0.15 0.2")

right_arm_joint, right_arm_geom = create_body_part(name="right_arm",
                                                   type="hinge",
                                                   size=f"{0.2} {0.04}", # length of right arm can be adjusted here
                                                   axis="-x",
                                                   pos="-0.2 0 1.6")

left_arm_joint, left_arm_geom = create_body_part(name="left_arm",
                                                 type="hinge",
                                                 size=f"{0.2} {0.04}", # length of left arm can be adjusted here
                                                 axis="-x",
                                                 pos="0.2 0 1.6")

right_leg_joint, right_leg_geom = create_body_part(name="right_leg",
                                                   type="hinge",
                                                   size=f"{0.4} {0.04}",
                                                   axis="-y,-x,-z,-y,-x,-z")

left_leg_joint, left_leg_geom = create_body_part(name="left_leg",
                                                 type="hinge",
                                                 size=f"{0.4} {0.04}",
                                                 axis="-y,x,z,-y,x,z")


# Appending body parts to body element
body.appendChild(head_joint)
body.appendChild(head_geom)
body.appendChild(torso_joint)
body.appendChild(torso_geom)
body.appendChild(right_arm_joint)
body.appendChild(right_arm_geom)
body.appendChild(left_arm_joint)
body.appendChild(left_arm_geom)
body.appendChild(right_leg_joint)
body.appendChild(right_leg_geom)
body.appendChild(left_leg_joint)
body.appendChild(left_leg_geom)

# Appending ground and body to worldbody element
worldbody.appendChild(ground)
worldbody.appendChild(body)

# Appending worldbody to root element
root.appendChild(worldbody)

# Writing the xml document to a file named humanoid.xml
xml_str = root.toprettyxml(indent="\t") # Formatting the xml string with indentation
with open('example.xml', 'w') as f:
    f.write(xml_str)