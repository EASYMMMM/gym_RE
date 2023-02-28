'''
hello pybullet world
python learning/hellopybullet.py
'''
import pybullet as p
import time
import pybullet_data
# 连接物理引擎  GUI：带可视化界面   DIRECT：不带可视化界面，直接连接物理引擎
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
# 添加data路径 
# ../anaconda3/envs/my_pybulletLearning/lib/python3.9/site-packages/pybullet_data
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally

p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # 关闭界面两侧控制窗口
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0) # 关闭渲染界面，等场景load完成后再开启
# 设置重力
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0,0,1]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
boxId = p.loadURDF("r2d2.urdf",cubeStartPos, cubeStartOrientation)

# 添加双足机器人
bipedStartPos = [2,2,2]
bipedStartOrientation = p.getQuaternionFromEuler([0,0,0])
bipedId = p.loadURDF("biped/biped2d_pybullet.urdf",bipedStartPos,bipedStartOrientation)

# 开启渲染
print("=" * 20)
print("渲染加载完成")
print("=" * 20)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1) 

for i in range (5000):
    # 离散步模拟
    p.stepSimulation()
    # 时间间隔为1/240时比较合理
    time.sleep(1./240.)
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)

print("=" * 20)
print(f"机器人的位置坐标为:{cubePos}\n机器人的朝向四元数为:{cubeOrn}")
print("=" * 20)
# 断开物理引擎连接
p.disconnect()