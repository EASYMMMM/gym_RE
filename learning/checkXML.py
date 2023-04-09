'''
使用chatGPT生成的检查xml文件的代码。
python learning/checkXML.py
'''
# ------- 来自于mujoco150在win+py3.9下的矫情的要求 --------
# 手动添加mujoco路径
import os
from getpass import getuser
import time
user_id = getuser()
os.add_dll_directory(f"C://Users//{user_id}//.mujoco//mujoco200//bin")
os.add_dll_directory(f"C://Users//{user_id}//.mujoco//mujoco-py-2.0.2.0//mujoco_py")
# -------------------------------------------------------
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from gym_custom_env.generateXML import HumanoidXML
import mujoco_py
import numpy as np
from mujoco_py.generated import const
from scipy.spatial.transform import Rotation as R

def quaternion_to_rotation_matrix(q):  # x, y ,z ,w
    rot_matrix = np.array(
        [[1.0 - 2 * (q[1] * q[1] + q[2] * q[2]), 2 * (q[0] * q[1] - q[3] * q[2]), 2 * (q[3] * q[1] + q[0] * q[2])],
         [2 * (q[0] * q[1] + q[3] * q[2]), 1.0 - 2 * (q[0] * q[0] + q[2] * q[2]), 2 * (q[1] * q[2] - q[3] * q[0])],
         [2 * (q[0] * q[2] - q[3] * q[1]), 2 * (q[1] * q[2] + q[3] * q[0]), 1.0 - 2 * (q[0] * q[0] + q[1] * q[1])]],
        dtype=q.dtype)
    return rot_matrix

# params list
paramas = { #'torso_width':0.5,
            #'init_position':[0,0,2.5],
            #'pelvis_width':0.2,
            'upper_arm_lenth':0.31,
            'lower_arm_lenth':0.4,
            #'shin_lenth':0.5,
            #'torso_height':0.6,
            }

# 生成XML文件
t = HumanoidXML(terrain_type='ladders',gravity=0)
t.write_xml(file_path="e.xml")


# 更新XML文件
""" t.set_params(paramas)
t.update_xml(file_path='e.xml') """

# 加载 XML 文件
# model = mujoco_py.load_model_from_path("gym_custom_env\\assets\\humanoid_custom.xml")
model = mujoco_py.load_model_from_path("e.xml")
# model = mujoco_py.load_model_from_path("humanoid.xml")
# 创建仿真环境和渲染器
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

# 设置仿真时间步长和仿真时长
dt = 0.01
timesteps = 500000
viewer.render()

ctrl = np.zeros(len(sim.data.ctrl[:]))

#ctrl[13] =  0.050 # right shoulder 1
#ctrl[16] =  0.050 # left shoulder 1

sim.data.ctrl[:] = ctrl
j = 0
k = 0

geomdict = {}
for i in range(sim.model.ngeom):
    geomdict[i] = sim.model.geom_id2name(i)
print(geomdict)


# 运行仿真并在每个时间步骤中进行渲染
for i in range(timesteps):
    sim.step()
    viewer.render()

    if i % 1000 == 0:
        if j == 0:
            #ctrl[3] = -0.050 # right hip x
            #ctrl[4] =  -0.050 # right hip z  
            #ctrl[5] =  -0.050 # right hip y            
            #ctrl[6] =  -0.050 # right knee
            #ctrl[8] = -0.050 # left hip x
            #ctrl[9] =  -0.050 # left hip z  
            #ctrl[10] =  -0.050 # left hip y  
            #ctrl[11] = -0.050 # left knee           
            #ctrl[7] = -0.050 # right ankle
            #ctrl[12] = 0.050 # left ankle
            ctrl[13] = -0.050 # right shoulder 1
            #ctrl[14] = -0.050 # right shoulder 2
            ctrl[15] = -0.050 # right elbow
            ctrl[16] = -0.050 # left shoulder 1
            #ctrl[17] = -0.050 # left shoulder 2
            ctrl[18] = -0.050 # left elbow             
            sim.data.ctrl[:] = ctrl
            j = 1
        else:
            #ctrl[3] =  0.050 # right hip x  
            #ctrl[4] =  0.050 # right hip z  
            #ctrl[5] =  0.050 # right hip y                       
            #ctrl[6] =  0.050 # right knee
            #ctrl[7] = 0.050 # right ankle
            #ctrl[8] =  0.050 # left hip x
            #ctrl[9] =  0.050 # left hip z  
            #ctrl[10] =  0.050 # left hip y   
            #ctrl[11] =  0.050 # left knee            
            #ctrl[12] = -0.05 # left ankle
            ctrl[13] =  0.050 # right shoulder 1
            #ctrl[14] =  0.050 # right shoulder 2
            ctrl[15] =  0.050 # right elbow 
            ctrl[16] =  0.050 # left shoulder 1
            #ctrl[17] =  0.050 # left shoulder 2
            ctrl[18] =  0.050 # left elbow               
            sim.data.ctrl[:] = ctrl
            j = 0            

    torso_z = sim.data.qpos[2]
    #print(torso_z)
    #viewer.add_marker(pos=[0,0,torso_z], size=np.array([0.05, 0.05, 0.05]), rgba=np.array([1.0, 0, 0.0, 1]), type=const.GEOM_SPHERE)

    torso_x = sim.data.qpos[0]
    #print(torso_x)
    #viewer.add_marker(pos=[torso_x,1,torso_z], size=np.array([0.05, 0.05, 0.05]), rgba=np.array([1.0, 0, 0.0, 1]), type=const.GEOM_SPHERE)


    #viewer.add_marker(pos=[0,-0.14,0.4], size=np.array([0.1, 0.1, 0.1]), rgba=np.array([0, 0, 1.0, 1]), type=const.GEOM_SPHERE)
    #viewer.add_marker(pos=[-1,0,1.7], size=np.array([0.01, 0.01, 0.01]), rgba=np.array([0, 0, 1.0, 1]), type=const.GEOM_SPHERE)
    #viewer.add_marker(pos=[0,0,0.4], size=np.array([0.1, 0.1, 0.1]), rgba=np.array([0, 0, 1.0, 1]), type=const.GEOM_SPHERE)
    quatanion = sim.data.qpos[3:7]
    Rm = R.from_quat(quatanion)  # Rotation matrix
    #rotation_matrix = Rm.as_matrix()


    rotation_matrix=quaternion_to_rotation_matrix(quatanion)

    vertical_direction = np.array([0, 0, 1])
    body_z_axis = rotation_matrix.dot(vertical_direction)
    dot_product = np.dot(body_z_axis, vertical_direction)
'''
    # 读取mujoco碰撞参数
    contact = list(sim.data.contact)
    print('==================================')
    print('geom number: ', sim.model.ngeom)
    print('number of detected contacts:',sim.data.ncon)
    #print('geom name floor id:' , sim.model.geom_name2id("floor"))
    #print('geom name lwaist id:' , sim.model.geom_name2id("lwaist"))
    print('geom1 id:',contact[1].geom1,' geom1 name:',sim.model.name_geomadr[contact[1].geom1])
    print('geom2 id:',contact[1].geom2,' geom2 name:',sim.model.name_geomadr[contact[1].geom2])
    print('  ')
    '''
# 关闭仿真环境和渲染器
viewer.close()
sim.close()

