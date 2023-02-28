'''
from 知乎
机器人控制测试
python learning/robotControlTest.py
'''
import pybullet as p
import pybullet_data
import time 
from pprint import pprint

# 连接物理引擎
use_gui = True
if use_gui:
    serve_id = p.connect(p.GUI)
else:
    serve_id = p.connect(p.DIRECT)

# 添加资源路径
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 配置渲染机制
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

# 设置重力，加载模型
p.setGravity(0, 0, -10)
_ = p.loadURDF("plane.urdf", useMaximalCoordinates=True)
robot_id = p.loadURDF("r2d2.urdf", useMaximalCoordinates=True)

# 可以使用的关节
available_joints_indexes = [i for i in range(p.getNumJoints(robot_id)) if p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED]

print('='*30)
print('r2d2可以控制的关节:')
pprint([p.getJointInfo(robot_id, i)[1] for i in available_joints_indexes])
print('='*30)

# 获取轮子的关节索引
wheel_joints_indexes = [i for i in available_joints_indexes if "wheel" in str(p.getJointInfo(robot_id, i)[1])]
print('='*30)
print('轮子的关节索引:')
print(wheel_joints_indexes)
print('='*30)

# 预备工作结束，重新开启渲染
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
# 关闭实时模拟步
p.setRealTimeSimulation(0)

target_v = 10                   # 电机达到的预定角速度（rad/s）
max_force = 10                  # 电机能够提供的力，这个值决定了机器人运动时的加速度，太快会翻车哟，单位N

print('='*30)
print([target_v for _ in wheel_joints_indexes])
print([max_force for _ in wheel_joints_indexes])
print('='*30)


for i in range(1000):
    p.stepSimulation()
    p.setJointMotorControlArray(
        bodyUniqueId=robot_id,
        jointIndices=wheel_joints_indexes,
        controlMode=p.VELOCITY_CONTROL,
        #targetVelocities=[target_v for _ in wheel_joints_indexes],
        targetVelocities=[10,10,-10,-10],  # 输出速度序列 原地旋转
        forces=[max_force for _ in wheel_joints_indexes]
    )
    # 获得机器人baselink的坐标和四元数
    location, _ = p.getBasePositionAndOrientation(robot_id)
    p.resetDebugVisualizerCamera(
        cameraDistance=3,
        cameraYaw=110,
        cameraPitch=-30,
        cameraTargetPosition=location
    )
    time.sleep(1 / 240)         # 模拟器一秒模拟迭代240步

# 断开连接
p.disconnect(serve_id)