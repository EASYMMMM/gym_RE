'''
collisionTest
碰撞检测测试
https://zhuanlan.zhihu.com/p/347739508

python learning/collisionTest.py

'''
import pybullet as p
import pybullet_data
from time import sleep

use_gui = True
move = False
fly = True

if use_gui:
    serve_id = p.connect(p.GUI)
else:
    serve_id = p.connect(p.DIRECT)

# 配置渲染机制
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

# 添加资源路径
p.setAdditionalSearchPath(pybullet_data.getDataPath())
_ = p.loadURDF("plane.urdf", useMaximalCoordinates=True)
robot_id = p.loadURDF("r2d2.urdf", basePosition=[0, 0, 0.5], useMaximalCoordinates=True)

# 创建一面墙
visual_shape_id = p.createVisualShape(
    shapeType=p.GEOM_BOX,
    halfExtents=[60, 2, 2]
)

collison_box_id = p.createCollisionShape(
    shapeType=p.GEOM_BOX,
    halfExtents=[60, 2, 2]
)

wall_id = p.createMultiBody(
    baseMass=10000,
    baseCollisionShapeIndex=collison_box_id,
    baseVisualShapeIndex=visual_shape_id,
    basePosition=[0, 10, 5]
)

# 重新开始渲染，开启重力
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.setGravity(0, 0, -10)
p.setRealTimeSimulation(0)

if move:
    for i in range(p.getNumJoints(robot_id)):
        if "wheel" in p.getJointInfo(robot_id, i)[1].decode("utf-8"):  #如果是电机         # 如果是轮子的关节，则为马达配置参数，否则禁用马达
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=i,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=30,
                force=100
            )
        else:
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=i,
                controlMode=p.VELOCITY_CONTROL,
                force=0
            )

if fly:
    for _ in range(10):
        p.applyExternalForce(
            objectUniqueId=robot_id,
            linkIndex=-1,
            forceObj=[-1, 10000, 12000],
            posObj=p.getBasePositionAndOrientation(robot_id)[0],
            flags=p.WORLD_FRAME
        )
        p.stepSimulation()
        sleep(1 / 240)

while True:
    p.stepSimulation()
    P_min, P_max = p.getAABB(robot_id)
    id_tuple = p.getOverlappingObjects(P_min, P_max)
    if len(id_tuple) > 1:  #总会认为机器人自己和自己碰撞
        for ID, _ in id_tuple:
            if ID == robot_id:
                continue
            else:
                print(f"hit happen! hit object is {p.getBodyInfo(ID)}")
    sleep(1 / 240)


p.disconnect(serve_id)