'''
使用chatGPT生成的检查xml文件的代码。
python learning/checkXML.py
'''
# ------- 来自于mujoco150在win+py3.9下的矫情的要求 --------
# 手动添加mujoco路径
import os
from getpass import getuser
user_id = getuser()
os.add_dll_directory(f"C://Users//{user_id}//.mujoco//mujoco200//bin")
os.add_dll_directory(f"C://Users//{user_id}//.mujoco//mujoco-py-2.0.2.0//mujoco_py")
# -------------------------------------------------------

import mujoco_py

# 加载 XML 文件
# model = mujoco_py.load_model_from_path("gym_custom_env\\assets\\humanoid_custom.xml")
model = mujoco_py.load_model_from_path("humanoid.xml")

# 创建仿真环境和渲染器
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

# 设置仿真时间步长和仿真时长
dt = 0.01
timesteps = 5000
viewer.render()
# 运行仿真并在每个时间步骤中进行渲染
for i in range(timesteps):
    sim.step()
    viewer.render()

# 关闭仿真环境和渲染器
viewer.close()
sim.close()
