'''
from 知乎
有关机器人信息的查询函数
python learning/robotCheckTest.py
'''

import pybullet as p
import pybullet_data
from time import sleep


# 链接物理引擎
p.connect(p.GUI)

# 设置模型加载路径
datapath = pybullet_data.getDataPath()
p.setAdditionalSearchPath(datapath)

# 配置渲染机制
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

# 设置重力，加载模型
p.setGravity(0, 0, -10)

# 加载模型
_ = p.loadURDF("plane.urdf", useMaximalCoordinates=True)
#robot_id = p.loadURDF("r2d2.urdf", [0, 0, 0.5])
#robot_id = p.loadURDF("./gym_custom_env/miniBox.urdf",[0,0,1])
robot_id = p.loadURDF("./gym_custom_env/atlas/atlas_v4_with_multisense.urdf",[0,0,2])
# 预备工作结束，重新开启渲染
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
# 关闭实时模拟步
p.setRealTimeSimulation(0)

# 输出基本信息
joint_num = p.getNumJoints(robot_id)
print("robot的节点数量为：", joint_num)

print("robot的信息：")
for joint_index in range(joint_num):
    info_tuple = p.getJointInfo(robot_id, joint_index)
    print("="*20)
    print(f"关节序号：{info_tuple[0]}\n\
            关节名称：{info_tuple[1]}\n\
            关节类型：{info_tuple[2]}\n\
            机器人第一个位置的变量索引：{info_tuple[3]}\n\
            机器人第一个速度的变量索引：{info_tuple[4]}\n\
            保留参数：{info_tuple[5]}\n\
            关节的阻尼大小：{info_tuple[6]}\n\
            关节的摩擦系数：{info_tuple[7]}\n\
            slider和revolute(hinge)类型的位移最小值：{info_tuple[8]}\n\
            slider和revolute(hinge)类型的位移最大值：{info_tuple[9]}\n\
            关节驱动的最大值：{info_tuple[10]}\n\
            关节的最大速度：{info_tuple[11]}\n\
            节点名称：{info_tuple[12]}\n\
            局部框架中的关节轴系：{info_tuple[13]}\n\
            父节点frame的关节位置：{info_tuple[14]}\n\
            父节点frame的关节方向：{info_tuple[15]}\n\
            父节点的索引，若是基座返回-1：{info_tuple[16]}\n\n")

while True:
    p.stepSimulation()
    sleep(1. / 240.)