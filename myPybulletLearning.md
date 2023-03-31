



# Pybullet+gym+pytorch+stable baselines 学习笔记

## 1. Pybullet

[pybullet官方文档说明](https://usermanual.wiki/Document/pybullet20quickstart20guide.479068914.pdf)

### 一些pybullet例程

- **鼠标互动足球**
  ` python -m pybullet_examples.soccerball`
- **一个简单的双足机器人模型**
  ` python -m pybullet_examples.biped2d_pybullet`
- **bipedal-robot-walking-simulation-master**
  一个简单的控制双足机器人行走的例程。自行设计简单步态，根据步态完成各个关节轨迹规划，PD控制驱动关节。
- **robotCheckTest**
  有关机器人信息的查询函数。
- **robotControlTest**
  机器人控制测试。r2d2机器人。
- **duckTest**
  通过3d文件生成碰撞体测试。自带的小黄鸭模型。
- **collisionTest**
  AABB包围箱，碰撞测试。
- **debugTest**
  在GUI上添加交互按钮，添加说明文字等。

###  pybullet一些常用API：

```python
import pybullet as p
```

- 连接物理引擎  GUI：带可视化界面   DIRECT：不带可视化界面，直接连接物理引擎
  `physicsClient = p.connect(p.GUI)`
- 添加模型搜索路径
  `p.setAdditionalSearchPath(pybullet_data.getDataPath()) `
- 关闭界面两侧控制窗口
  ` p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  `
- 关闭/开启渲染界面
  `p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0) `
- 设置重力
  `p.setGravity(0,0,-10)`
- 加载机器人 参数：filename, base position, base orientation, useMaximalCoordinates(选用后会提升仿真速度), use fixed base, flags(详见pdf), global scaling, physics client ID
  ` boxId = p.loadURDF("r2d2.urdf",cubeStartPos, cubeStartOrientation)`
- 获取机器人关节数目，返回关节数目的int值
  ` p.getNumJoints(robot, physicsClientId=physicsClientId)`
- 获取关节信息 
  `  jointInfo = p.getJointInfo(self._robot, i, physicsClientId=self._physicsClientId)`

##  2. 虚拟环境

### linux环境

- **环境：conda虚拟环境 python3.9**
- **创建虚拟环境**
    ` conda create -n my_pybulletLearning python=3.9 `
    ` conda activate my_pybulletLearning`
    更新pip
    ` pip install --upgrade pip`
    更换清华源
    ` pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple`
- **安装pybullet及其他需要的包**
    ` pip install pybullet`
    ` pip install numpy==1.19.3` 
    ` pip install gym==0.21.0` 

###  windows环境

- **环境：conda虚拟环境 python3.9**
- **创建虚拟环境**
    windows下使用conda的配置更复杂一些，需要配置环境变量。百度即可。
    同时记得百度一下怎么将虚拟环境的默认安装地址从c盘改到其他盘。
    ` conda create -n my_pybulletLearning python=3.9 `
    ` conda activate my_pybulletLearning`
    更新pip
    ` conda install pip`
    ` pip install --upgrade pip`
    更换清华源
    ` pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple`
- 安装pybullet及其他包
    ` conda install pybullet`
    ` conda install gym=0.21.0`
    ` conda install pyglet`

## 3. Mujoco

**[Mujoco官网](https://mujoco.readthedocs.io/en/latest/overview.html)**

**[Mujoco py官网](https://openai.github.io/mujoco-py/build/html/index.html)**

**[colab教程](https://colab.research.google.com/github/deepmind/mujoco/blob/main/python/tutorial.ipynb#scrollTo=-P95E-QHizQq)**

### 安装mujoco

- **win**
  在win11系统下安装mujoco。 

  由于mujoco官方已经放弃win：

  >  Windows support has been DEPRECATED and removed in [2.0.2.0](https://github.com/openai/mujoco-py/releases/tag/v2.0.2.0a1). One known good past version is [1.50.1.68]

  虽然他说放弃了，但仍然选择安装mujoco200 + mujoco-py 2.0.2.0。 网络上有不少教程是关于win安装mujoco200。亲测还能用。

  需要下载visual studio build tools，我的版本是2022，目前可以运行。 在[ms官网](https://visualstudio.microsoft.com/zh-hans/downloads/) 下载installer，安装`visual studio 2022生成工具`。  

  在[mujoco官网](https://www.roboti.us/download.html)上下载mujoco安装包，选择`mujoco200 win64`。随后在user文件夹下创建` .mujoco`文件夹，在这个文件夹里解压压缩包，并配置环境变量。[具体过程参照这篇](https://blog.csdn.net/Sunctam/article/details/124354051)。配置环境变量后若仍提示找不到key文件，重启一下即可。   

  安装mujoco-py时，从[github](https://github.com/openai/mujoco-py)上下载mujoco-py压缩包。版本选择`2.0.2.0`。建议将mujoco-py的文件夹也放到.mujoco下面，后续会用到。解压后，将文件夹`.mujoco\mujoco-py-2.0.2.0\`内的`mujoco-py\`文件夹拷贝到虚拟环境储存安装包的路径下。我的路径为`E:\anaconda3\envs\GYM\Lib\site-packages`。

  随后打开anaonda prompt终端，先定位到从github上下载的压缩包。

  ` cd C:\Users\xxx\.mujoco\mujoco-py-2.0.2.0 ` 

  ` pip install -r requirements.txt  `   

  ` pip install -r requirements.dev.txt  ` 

  ` python setup.py install`

  我参考的教程上执行完这一步后，在` conda list `里就可以看到mujoco-py了。  

  测试代码：

  ```python
  import mujoco_py
  import os
  mj_path, _ = mujoco_py.utils.discover_mujoco()
  xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
  model = mujoco_py.load_model_from_path(xml_path)
  sim = mujoco_py.MjSim(model)
  print(sim.data.qpos)
  #[0.  0.  1.4 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  # 0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ]
  sim.step()
  print(sim.data.qpos)
  #[-1.12164337e-05  7.29847036e-22  1.39975300e+00  9.99999999e-01
  #  1.80085466e-21  4.45933954e-05 -2.70143345e-20  1.30126513e-19
  # -4.63561234e-05 -1.88020744e-20 -2.24492958e-06  4.79357124e-05
  # -6.38208396e-04 -1.61130312e-03 -1.37554006e-03  5.54173825e-05
  # -2.24492958e-06  4.79357124e-05 -6.38208396e-04 -1.61130312e-03
  # -1.37554006e-03 -5.54173825e-05 -5.73572648e-05  7.63833991e-05
  # -2.12765194e-05  5.73572648e-05 -7.63833991e-05 -2.12765194e-05]
  ```

    在` import mujoco_py`时有可能会报错：

    ```shell
    ...
      File "E:\ANACONDA\envs\GYM\lib\site-packages\mujoco_py\builder.py", line 55, in load_cython_ext
        mod = imp.load_dynamic("cymj", cext_so_path)
      File "E:\ANACONDA\envs\GYM\lib\imp.py", line 342, in load_dynamic
        return _load(spec)
  ImportError: DLL load failed while importing cymj: 找不到指定的模块。
    ```

    网上有教程说这是python版本太高导致的问题。解决方法为，在调用mujoco-py前手动添加mujoco路径。xxx为你的用户名。

  ```python
  import os
  from getpass import getuser
  user_id = getuser()
  os.add_dll_directory(f"C://Users//{user_id}//.mujoco//mujoco200//bin")
  os.add_dll_directory(f"C://Users//{user_id}//.mujoco//mujoco-py-2.0.2.0//mujoco_py")
  ```

    参考：https://blog.csdn.net/alan1ly/article/details/126087866 

    参考：https://zhuanlan.zhihu.com/p/502112539

- **linux**
  mujoco安装210

  [Mujoco210和Mujoco-py的安装 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/486957504)     

  [openai/mujoco-py: MuJoCo is a physics engine for detailed, efficient rigid body simulations with contacts. mujoco-py allows using MuJoCo from Python 3. (github.com)](https://github.com/openai/mujoco-py)

  第二个链接里包含了mujoco和mujoco-py的安装 先根据教程安装mujoco210 并在bashrc中添加这几个 具体根据实际路径就行

  有个教程里说要加什么nVidia不知道干吗用 我没加但也能正常使用

  ```
  export MUJOCO_KEY_PATH=/home/sam/.mujoco${MUJOCO_KEY_PATH}
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sam/.mujoco/mujoco210/bin
  export MUJOCO_PY_MUJOCO_PATH=/home/sam/.mujoco/mujoco210
  ```

  然后pip3 install -U 'mujoco-py<2.2,>=2.1'

##  4. gym

**gym版本目前暂时选用0.21.0，发布于2021年10月。**  

目前最新版本的gym为0.26.2，相比于0.21.0更改了许多关键API，新旧版本并不兼容。使用pip安装时记得在anaconda prompt中使用。

` pip install gym==0.21.0 `     

安装完成后测试一下来自pybullet的官方例程，能看到一个踉跄奔跑的小人。  

` python -m pybullet_envs.examples.enjoy_TF_HumanoidBulletEnv_v0_2017may`     

打印gym所有可用env：

```python
from gym import envs
for env in envs.registry.all():
    print(env.id)
```
一个适用于gym 0.21.0版本的例程：

```python
import gym
print(gym.__version__)
env = gym.make('CartPole-v1')  #倒立摆
env.reset()
for _ in range(10000):
    env.render()                            # 渲染
    act = env.action_space.sample()         # 在动作空间中随机采样
    obs, reward, done, _ = env.step(act)    # 与环境交互
    if done:
        env.reset()
env.close()	
```

gym的一些例程：

- **gymTest0**
  gym 测试，生成一个倒立摆
- **gymTest1**
  尝试通过pybullet引入gym环境，生成一个倒立摆
- **[CSDN] gym搭建自己的环境，全网最详细版本，3分钟你就学会了！**
https://blog.csdn.net/sinat_39620217/article/details/115519622

### Space类

`gym.space`类定义观察和操作空间，因此您可以编写适用于任何Env的通用代码。

用作observation_space和action_space。

### Env类

所有的gym强化学习环境，都是`gym.env`的派生类。定义在gym/core.py文件中。

**Env类的主要函数有：**

- **step**()

  运行仿真的一个时间步长。当一个episode结束时，需要调用`reset()`来重置这个环境的状态。

  输入参数：

  - action (object): 智能体需要执行的操作

  返回参数：

  - observation (object): 智能体在当前状态下的观测值。
  - reward (float) : 当前奖励值
  - done (bool): 当前episode是否结束。未结束时，继续调用`setp()`；若结束则下一个循环调用`reset()`。
  - info (dict): 包含辅助诊断信息(有助于调试，有时还有助于学习)。

- **reset()**

  将环境重置为初始状态并返回初始观测值。    Note that this function should not reset the environment's random number generator(s); random variables in the environment's state should be sampled independently between multiple calls to `reset()`. In other words, each call of `reset()` should yield an environment suitable for a new episode, independent of previous episodes.

  输入参数：None

  返回参数：

  - observation (object): 初始的观测值。

- **render()**

  渲染环境图像。

  输入参数：

  - mode (str): 渲染模式。

    mode = human: 渲染到当前显示器或终端，不返回任何内容。供观看。
    mode = rgb_array: 返回numpy类型。形状为(x, y, 3)的np.ndarray，表示x-by-y像素图像的RGB值，适合转换为视频。

    mode = ansi: 返回字符串(str)或StringIO。包含终端样式文本表示的StringIO。文本可以包括换行符和ANSI转义序列(例如用于颜色)。

- **seed()**

  设定env环境的随机数生成器的种子。

### 自定义Env环境

自定义用户env环境时，文件结构需要满足如下要求：

```
gym-basic/
  README.md
  setup.py
  gym_basic/
    __init__.py
    envs/
      __init__.py
      basic_env.py
      basic_env_2.py
```





##  5. stable_baselines

安装stablebaselines3，同时安装pyrender用于图像展示。
` conda install stable_baselines3`
` conda install pyrender` 
或者从conda安装，可能有点慢
` conda install -c conda-forge stable-baselines3`

或直接pip安装，记得在anaconda prompt里使用pip

` pip install stable-baselines3==1.5.0`

**版本暂时选用1.5.0，该版本要求gym>=0.21.0。**

什么是stable_baselines？
基于pytorch的深度强化学习工具包，可直接联合gym，对gym的env环境进行训练。

- Stable Baselines3 (SB3) is a set of reliable implementations of reinforcement learning algorithms in PyTorch. It is the next major version of Stable Baselines.
- [stable_baselines官方文档](https://stable-baselines3.readthedocs.io/en/master/index.html)
- 【CSDN】一篇比较详细的介绍gym和stable baselines3的文档
  [强化学习之stable_baseline3详细说明和各项功能的使用](https://blog.csdn.net/tianjuewudi/article/details/123113885)

### **来自pybullet官方文档的例程**

- pybullet官方文档里提供了使用stable_baselines的例程。  
  ` pip3 install stable_baselines --user`
  训练一个HalfCheetah（半拉猎豹）智能体：
  ` python -m pybullet_envs.stable_baselines.train --algo sac --env HalfCheetahBulletEnv-v0`
  有可能会报错，需要再手动安装stable_baselines3
  ` pip3 install stable_baselines3`
  默认迭代次数为1e6，我花了二十分钟才跑了十分之一，直接ctrl+C中断了。中断后会自动保存训练好的参数，训练参数结果会保存在环境同名文件夹下（sac_HalfCheetahBulletEnv-v0）。
- 通过pybullet_envs/stable_baselines文件夹下的enjoy.py文件来查看训练结果。我训练了20分钟，十一万次迭代，效果比较差，猎豹智能体基本站不起来。
  ` a`
- **尝试训练新的环境**
    训练gym中HumanoidBulletEnv-v0的环境
    ` python -m pybullet_envs.stable_baselines.train --algo sac --env HumanoidBulletEnv-v0`
    ` python -m pybullet_envs.stable_baselines.enjoy --algo sac --env HumanoidBulletEnv-v0 --n-episodes 5`  
    在调试的.json文件中添加args：
    ` "args": ["--algo", "sac", "--env", "HumanoidBulletEnv-v0"],`    

### 一些测试用例

- **来自官方文档**  

  使用A2C训练倒立摆

  ```python
  import gym
  
  from stable_baselines3 import A2C
  
  env = gym.make("CartPole-v1")  #调用env
  
  model = A2C("MlpPolicy", env, verbose=1)  #定义policy
  model.learn(total_timesteps=10_000)       #训练，timesteps=10000
  
  vec_env = model.get_env()
  obs = vec_env.reset()
  for i in range(1000):
      action, _state = model.predict(obs, deterministic=True)
      obs, reward, done, info = vec_env.step(action)
      vec_env.render()
      # VecEnv resets automatically
      # if done:
      #   obs = vec_env.reset()
  ```

  或者，只用一行代码来训练已经注册好的模型：

  ```python
  from stable_baselines3 import A2C
  
  model = A2C("MlpPolicy", "CartPole-v1").learn(10000)
  ```

  







##  6. GPU加速pytorch

为了使用GPU加速的pytorch，单独创建一个虚拟环境。  

` conda create -n torch_gpu python=3.9`
` conda activate python=3.9`  

重新安装相关包：
` conda install pybullet`
` conda install gym=0.21.0`
` conda insatll pyglet`
` conda install stable-baselines3`  

安装不上时可以试试    

` conda install -c conda-forge stable-baselines3`
` conda install pyrender`
GPU加速pytorch安装步骤：
[pytorch gpu版本的最全安装教程，含环境配置、CUDA（百度云下载）安装程序失败解决方案](https://blog.csdn.net/L1778586311/article/details/112425993)
安装pytorch时参考下文，否则很容易用conda的镜像安装成cpu版本：
[torch.cuda.is_available()返回false——解决办法](https://blog.csdn.net/qq_46126258/article/details/112708781)    
CUDA11.6对应版本的pytorch安装：
` pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116`
安装完pytorch后，在命令行输入：

```shell
python
import torch
print(torch.__version__)
```
如果运行结果为` 1.7.1+cpu`或其他cpu版本，立刻从头再来...
安装完CUDA并配置好环境变量后，命令行输入：

```shell
python
import torch
torch.cuda.is_available()
```
如果结果为`True`，则安装完成。
运行测试用例`python learning/sb3Test0.py`,可以看到第一行打印为：
`Using cuda device`



## 7. 强化学习

### 问题记录

- [ ] 强化学习越训练越慢？训练一段时间后，同样的time steps耗时明显变长。

- [x] 同样的模型和代码，在笔记本3050ti上训练耗时32分钟，gpu占用率接近100。在桌面端TITAN RTX上训练耗时20分钟，gpu占用率（cuda占用率）不到50。

  多开几个ENV环境，能够有效解决这个问题。

## 8. tensorboard

  ` tensorboard --logdir`