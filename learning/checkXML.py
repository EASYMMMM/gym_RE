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
from collections import deque

def quaternion_to_rotation_matrix(q):  # x, y ,z ,w
    rot_matrix = np.array(
        [[1.0 - 2 * (q[1] * q[1] + q[2] * q[2]), 2 * (q[0] * q[1] - q[3] * q[2]), 2 * (q[3] * q[1] + q[0] * q[2])],
         [2 * (q[0] * q[1] + q[3] * q[2]), 1.0 - 2 * (q[0] * q[0] + q[2] * q[2]), 2 * (q[1] * q[2] - q[3] * q[0])],
         [2 * (q[0] * q[2] - q[3] * q[1]), 2 * (q[1] * q[2] + q[3] * q[0]), 1.0 - 2 * (q[0] * q[0] + q[1] * q[1])]],
        dtype=q.dtype)
    return rot_matrix

# 自定义参数
params = { #'torso_width':0.5,
            'thigh_lenth':0.34 * 1.05,            # 大腿长 0.34
            'thigh_size':0.06,             # 大腿粗 0.06
            'shin_lenth':0.3 * 0.7,              # 小腿长 0.3
            'shin_size':0.05,              # 小腿粗 0.05
            'upper_arm_lenth':0.2771 * 1.22,      # 大臂长 0.2771
            'upper_arm_size':0.04,         # 大臂粗 0.04
            'lower_arm_lenth':0.2944 * 0.98,      # 小臂长 0.2944
            'lower_arm_size':0.031,        # 小臂粗 0.031
            'foot_lenth':0.18 * 1.21,             # 脚长   0.18
            }


# 生成XML文件
t_type = 'default'
t_type = 'ladders'
t = HumanoidXML(terrain_type=t_type,gravity=0)
t.write_xml(file_path="ee.xml")

params = {}
# STEPS evo_punish_s1
params = {   'thigh_lenth':0.3469,           # 大腿长 0.34
            'shin_lenth':0.2828,              # 小腿长 0.3
            'upper_arm_lenth':0.2775,        # 大臂长 0.2771
            'lower_arm_lenth':0.3234,        # 小臂长 0.2944
            'foot_lenth':0.1725,       }     # 脚长   0.18
# FLAT FLOOR evo_punish_s3
params = {   'thigh_lenth':0.3185,           # 大腿长 0.34
            'shin_lenth':0.231,              # 小腿长 0.3
            'upper_arm_lenth':0.3095,        # 大臂长 0.2771
            'lower_arm_lenth':0.2214,        # 小臂长 0.2944
            'foot_lenth':0.1526,       }     # 脚长   0.18
params = {   'steps_height':0.10,     }      # 楼梯高度   0.18  
# FLAT FLOOR evo_punish_s3
params = {   'thigh_lenth':0.02,           # 大腿长 0.34
            'shin_lenth':0.02,              # 小腿长 0.3
            'upper_arm_lenth':0.2,        # 大臂长 0.2771
            'lower_arm_lenth':0.21,        # 小臂长 0.2944
            'foot_lenth':0.08,    
            'gravity':0,   }     # 脚长   0.18  
# 更新XML文件
t.set_params(params)
t.update_xml(file_path='ee.xml')

# 加载 XML 文件
# model = mujoco_py.load_model_from_path("gym_custom_env\\assets\\humanoid_custom.xml")
model = mujoco_py.load_model_from_path("ee.xml")
#model = mujoco_py.load_model_from_path("gym_custom_env/assets/humanoid_exp.xml")
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


def get_geom_idname(sim):
    geomdict = {}
    for i in range(sim.model.ngeom):
        geomdict[i] = sim.model.geom_id2name(i)
    return geomdict

geomdict = get_geom_idname(sim)

print(geomdict)

right_sensor_pos = deque(maxlen=600)
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
            #ctrl[13] = -0.050 # right shoulder 1
            #ctrl[14] = -0.050 # right shoulder 2
            #ctrl[15] = -0.050 # right elbow
            #ctrl[16] = -0.050 # right wrist
            #ctrl[17] = -0.050 # left shoulder 1
            #ctrl[18] = -0.050 # left shoulder 2
            #ctrl[19] = -0.050 # left elbow     
            #ctrl[20] = -0.050 # left wrist          
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
            #ctrl[13] =  0.050 # right shoulder 1
            #ctrl[14] =  0.050 # right shoulder 2
            #ctrl[15] =  0.050 # right elbow 
            #ctrl[16] =  0.050 # right wrist
            #ctrl[16] =  0.050 # left shoulder 1
            #ctrl[17] =  0.050 # left shoulder 2
            #ctrl[18] =  0.050 # left elbow 
            #ctrl[20] =  0.050 # left wrist   
            sim.data.ctrl[:] = ctrl
            j = 0            

    torso_z = sim.data.qpos[2]
    #print(torso_z)
    #viewer.add_marker(pos=[0,0,torso_z], size=np.array([0.05, 0.05, 0.05]), rgba=np.array([1.0, 0, 0.0, 1]), type=const.GEOM_SPHERE)

    torso_x = sim.data.qpos[0]
    #print(torso_x)
    #viewer.add_marker(pos=[torso_x,1,torso_z], size=np.array([0.05, 0.05, 0.05]), rgba=np.array([1.0, 0, 0.0, 1]), type=const.GEOM_SPHERE)

    y = sim.data.qpos[1]
    #viewer.add_marker(pos=[15,y,1.7], size=np.array([0.05, 0.05, 0.05]), rgba=np.array([0, 0, 1.0, 1]), type=const.GEOM_SPHERE)
    #viewer.add_marker(pos=[0,0,0.4], size=np.array([0.1, 0.1, 0.1]), rgba=np.array([0, 0, 1.0, 1]), type=const.GEOM_SPHERE)
    #viewer.add_marker(pos=[0.6,0,0.4], size=np.array([0.1, 0.1, 0.1]), rgba=np.array([0, 0, 1.0, 1]), type=const.GEOM_SPHERE)
    #viewer.add_marker(pos=[6,0,3], size=np.array([0.1, 0.1, 0.1]), rgba=np.array([0, 0, 1.0, 1]), type=const.GEOM_SPHERE)
    #viewer.add_marker(pos=[0,1.2,0.5], size=np.array([0.1, 0.1, 0.1]), rgba=np.array([0, 0, 1.0, 1]), type=const.GEOM_SPHERE)

    pelvis_x, pelvis_y, pelvis_z = sim.data.get_geom_xpos('pelvis_geom')
    # viewer.add_marker(pos=[pelvis_x,pelvis_y,3], size=np.array([0.1, 0.1, 0.1]), rgba=np.array([0, 0, 1.0, 1]), type=const.GEOM_SPHERE)
    # viewer.add_marker(pos=[pelvis_x,pelvis_y,1.4], size=np.array([0.1, 0.1, 0.1]), rgba=np.array([0, 0, 1.0, 1]), type=const.GEOM_SPHERE)
    # viewer.add_marker(pos=[pelvis_x,pelvis_y,1.0], size=np.array([0.1, 0.1, 0.1]), rgba=np.array([0, 0, 1.0, 1]), type=const.GEOM_SPHERE)

    if i % 4 == 0 and t_type == 'ladders':
        right_sensor_pos.append(np.array(sim.data.geom_xpos[45]))
    for com in right_sensor_pos:
        viewer.add_marker(pos=com, size=np.array([0.01, 0.01, 0.01]), rgba=np.array([1., 0, 0, 1]), type=const.GEOM_SPHERE)

    #viewer.add_marker(pos=[-1,0,1.7], size=np.array([0.01, 0.01, 0.01]), rgba=np.array([0, 0, 1.0, 1]), type=const.GEOM_SPHERE)
    #viewer.add_marker(pos=[0,0,0.4], size=np.array([0.1, 0.1, 0.1]), rgba=np.array([0, 0, 1.0, 1]), type=const.GEOM_SPHERE)
 
    quatanion = sim.data.qpos[3:7]
    Rm = R.from_quat(quatanion)  # Rotation matrix
    #rotation_matrix = Rm.as_matrix()
    rotation_matrix=quaternion_to_rotation_matrix(quatanion)
    vertical_direction = np.array([0, 0, 1])
    x_dir = np.array([1,0,0])
    y_dir = np.array([0,1,0])
    body_x_axis = rotation_matrix.dot(x_dir)
    body_y_axis = rotation_matrix.dot(y_dir)
    body_z_axis = rotation_matrix.dot(vertical_direction)
    dot_product = np.dot(body_z_axis, vertical_direction)
    forward_direction = np.array([1, 0, 0])
    x_dot_product = np.dot(body_x_axis, forward_direction)
    x_pos_r = 0 if (( x_dot_product + 1.0) / 2.0) > 0.8 else -1
    """     print('================')
    print('x:')
    print(body_x_axis)
    print('x product:')
    print(x_dot_product)
    print('================') """
    # 读取mujoco碰撞参数
    ncon = sim.data.ncon
    contact = list(sim.data.contact)  # 读取一个元素为mjContact的结构体数组
    for i in range(ncon):
        con = contact[i]
        if 'right_foot_geom_3' in geomdict[con.geom1]+geomdict[con.geom2]:
            print(geomdict[con.geom1] + geomdict[con.geom2])
        if 'ladders' in geomdict[con.geom1]+geomdict[con.geom2]:
            ladder = geomdict[con.geom1] if 'ladders' in geomdict[con.geom1] else geomdict[con.geom2]
            # 判断是手还是脚
            if 'hand' in geomdict[con.geom1]+geomdict[con.geom2]:
            # 区分左右手加分
                limb = 'right_hand' if 'right' in geomdict[con.geom1]+geomdict[con.geom2] else 'left_hand'
            elif 'foot' in geomdict[con.geom1]+geomdict[con.geom2]:
                limb = 'right_foot' if 'right' in geomdict[con.geom1]+geomdict[con.geom2] else 'left_foot'
            else: # 若非手脚，跳过
                continue
        else:
            if 'floor' in geomdict[con.geom1]+geomdict[con.geom2]:
                ladder = geomdict[con.geom1] if 'floor' in geomdict[con.geom1] else geomdict[con.geom2]
                # 判断是手还是脚
                if 'hand' in geomdict[con.geom1]+geomdict[con.geom2]:
                # 区分左右手加分
                    limb = 'right_hand' if 'right' in geomdict[con.geom1]+geomdict[con.geom2] else 'left_hand'
                elif 'foot' in geomdict[con.geom1]+geomdict[con.geom2]:
                    limb = 'right_foot' if 'right' in geomdict[con.geom1]+geomdict[con.geom2] else 'left_foot'
                else: # 若非手脚，跳过
                    continue
    print('==================================')
    print('geom number: ', sim.model.ngeom)
    print('number of detected contacts:',sim.data.ncon)
    print('quatanion:',quatanion)
    print('torso z:',torso_z)
    print('Rm',rotation_matrix)
    print('body_x_axis',x_pos_r)
    print('body_y_axis',body_y_axis)
    print('body_z_axis',body_z_axis)
    for i in range(ncon):
        print(f'contact : {geomdict[sim.data.contact[i].geom1]} + {geomdict[sim.data.contact[i].geom2]}')
    print('geom_xpos lenth: ',len(sim.data.geom_xpos))
    print('geom_xpos[1]: ',sim.data.geom_xpos[1])
    print('torso x point: ',sim.data.qpos[0])
    print('ladder pos:',t.ladder_positions)
    print('left_hand_sensor_geom:',sim.data.get_geom_xpos('left_hand_sensor_geom')[0])     
    # print(geomdict)
    print('*************')    
    #print('geom name floor id:' , sim.model.geom_name2id("floor"))
    #print('geom name lwaist id:' , sim.model.geom_name2id("lwaist"))
    #print('geom1 id:',contact[1].geom1,' geom1 name:',sim.model.name_geomadr[contact[1].geom1])
    #print('geom2 id:',contact[1].geom2,' geom2 name:',sim.model.name_geomadr[contact[1].geom2])
    print('  ')
  
# 关闭仿真环境和渲染器
viewer.close()
sim.close()

