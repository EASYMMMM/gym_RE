import pybullet as p
import numpy as np
import casadi as cs


import pybullet_data

# 定义仿真参数
dt = 0.01
T = 1.0
N = int(T / dt)

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally

# 加载人形机器人
robot = p.loadURDF("urdf/my_robot.urdf")

# 初始化状态
q0 = np.zeros(p.getNumJoints(robot))
dq0 = np.zeros(p.getNumJoints(robot))

# 定义目标位置
target_position = np.array([0.5, 0.5, 0.5])

# 定义状态方程
def dynamics(x, u):
    q = x[:p.getNumJoints(robot)]
    dq = x[p.getNumJoints(robot):]
    torques = u

    ddq = np.zeros(p.getNumJoints(robot))
    p.calculateInverseDynamics(robot, q, dq, ddq, torques)
    return np.hstack((dq, ddq))

# 定义MPC控制器
x = cs.MX.sym("x", 2 * p.getNumJoints(robot))
u = cs.MX.sym("u", p.getNumJoints(robot))

x_next = x + dt * dynamics(x, u)

q_next = x_next[:p.getNumJoints(robot)]

obj = cs.mtimes([(q_next - target_position).T, (q_next - target_position)])

nlp = {'x': x, 'p': u, 'f': obj}
solver = cs.nlpsol("solver", "ipopt", nlp)

# 开始仿真
for i in range(1000):

    print('###',i)
    
    # 计算MPC控制输入
    x0 = np.hstack((q0, dq0))
    sol = solver(x0=x0, p=np.zeros(p.getNumJoints(robot)))
    u_mpc = sol["p"][:, 0]

    # 应用控制输入到仿真环境
    p.applyExternalTorque(robot, -1, u_mpc, flags=p.WORLD_FRAME)
    p.stepSimulation()
