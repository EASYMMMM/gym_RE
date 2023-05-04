#  Undergraduate Thesis

  

## 1.  Introduction

### 1.1 Introduction

近些年，Robot Evolution是人工智能和机器人的交叉领域中非常受人关注的方向。得益于算力和算法的提升，人工智能得以用于机器人的设计和优化。已经有不少研究人员使用人工智能作为机器人研究的工具。其中，发展较为成熟的方向有机器视觉与机械臂的结合，强化学习控制机械臂完成特定任务等。大部分工作来自于仿真，一小部分工作实现了sim-to-real。

除此之外，还有一部分工作将人工智能与强化学习中的agent概念相结合，开创出了一个新的概念：具身智能。学者们研究这些具有实际物理结构（在仿真中）的agent通过神经网络或强化学习训练能够表现出怎样的智能。

### 1.2  Content

We are going to propose an algorithm that can jointly optimize the controller and morphology of a humanoid robot,  which enable to get a better design in specific task. 

In order to complete the optimization of the robot morphology, several robots with different design parameters need to be evaluated.  The simplest way to make it is to give all the possible design parameters, evaluate each one, and select the one that performs best. However, the robot we are going to optimize is a humanoid robot with more than 20 links, which has a enormous design space. Besides, the evaluation process requires that each robot has an optimal controller. It would be very time-consuming and laborious to design the controllers for each robot with different morphology parameters individually.

 Therefore, a co-optimization strategy of morphology and controller is developed. The controller is optimized at the same time as the morphology is optimized. In this way, the total time cost can be significantly reduced.

<img src="E:\CASIA\A-毕业设计\IMG\图片1.svg" alt="图片1" style="zoom: 33%;" />

The co-optimization can be divided into two parts: controller optimization and morphology optimization. The controller optimization will be the inner layer, using reinforcement learning; as well as the morphology optimization is the outer later, using heuristic search algorithm. The evaluation of the method will be done in a simulation environment.



Figure 2 shows the overall research framework of this thesis. We built the simulation platform based on Mujoco and Gym, use pytorch to build the reinforcement learning neural network, use genetic algorithm as the outer layer optimization algorithm, and constructed the cooperation-morphology optimization framework by ourselves. The details are explained later. 

The main content of the thesis comprises the two-layer co-optimization algorithms development and the simulation environment construction. The high-level optimization is the morphology optimization, and the low-level is the controller optimization. All experiments and verification of the algorithm will be conducted on the Mujoco simulation platform. The terrains include two types: flat floor, stairs. The prototype of the humanoid robot model used in the thesis was taken from the Mujoco simulation platform and packaged as an environment in *gym*, a platform commonly used in the field of reinforcement learning today.

### 1.3 研究意义

Through this thesis, we hope to come up with a joint optimization algorithm that can perform co-optimization of robot morphology and controller under specific tasks. As mentioned above, the design of robot is a very complicated project. Robot morphology and controller are two crucial parts of the robot, they complement each other and cooperate with each other. Only appropriate morphology design with excellent performance of the controller, the robot can complete the target task. 

However, both parts of the design work are complex and usually done separately. In the traditional robot development process, if the design structure is changed, the dynamics parameters of the robot will also change, which means that the controller will have to be modified. Therefore, the morphological structure of the robot is usually determined first, and then the controller is developed. Whether the proposed morphological structure is optimal for a particular task is generally judged by the experience of the engineer. Our algorithm will be an attempt to use a computer to jointly optimize the controller and morphology. Since the controller we use is based on reinforcement learning, it can be changed at any time, and we don't need to know any of the robot's dynamic parameters, as long as the robot model in the simulation is reasonable.

At the same time, our algorithm framework can be applied to a variety of robots. Given the right simulation platform and clear task, our proposed algorithm has the ability to optimize any robot. This will lay the groundwork for possible follow-up work. Some optimization tests and evaluations can also be carried out on the basis of our algorithm.

In addition, our thesis is also inspired by relevant researches in the field of "embodied intelligence". Some studies have shown that there is also potential intelligence in the morphology of agent. For example, to train a hand to grasp by reinforcement learning, a hand that looks like a human hand will definitely perform better than a hand that looks like a cat's paw. Thus, form optimization is also a way to generate intelligence. Our work will also be an attempt to optimize the form of a humanoid robot to see if it exhibits new capabilities for specific tasks.





### 1.4 Background and motivation

Robots are often required to work in complex terrains and environments. The robots that work in different environments requires the corresponding structure, including the actuators, the morphology, the size, etc. For example, a robot driving in rugged mountainous terrain needs more flexible joints to help it to cross obstacles and climb. Thus, good structural design is critical to a robot's ability to perform tasks in complex environments. On the other side, the manual design for robot structure is time consuming and requires necessary expertise.

Nowadays, there are more and more types and functions of robots, and bipedal robots and humanoid robots are developing rapidly. Compared with traditional wheeled and tracked robots, the design and control of bipedal robots are much more complex, requiring consideration of balance, gait and other issues. At the same time, the advantages of bipedal robots are also very prominent, as they are more adaptable, adapt to more diverse scenarios, have certain barrier-crossing capabilities, and can also perform multiple types of work.

<img src="E:\CASIA\A-毕业设计\IMG\atlas2016-1200x630.jpg" alt="atlas2016-1200x630" style="zoom:33%;" />



But designing such a robot is a very long process, including mechanical structure design, morphological parameters design, electrical system design, controller design, and many other parts. Each of these parts is very time-consuming. Therefore, we would like to build a simulation platform for automatic optimization of robot morphology and controllers to improve the efficiency of the robot design process. For example, the atlas robot in Figure 1, developed by Boston Dynamics, is a hydraulically driven, very powerful humanoid robot. Its first prototype was born in 1983 and has evolved over four decades into what you see today.

In this thesis, we propose an optimization method that can co-optimize the morphology and controller of a humanoid robot model under specific terrain and tasks, and finally get a better humanoid robot design. Our algorithm will optimize the robot morphology and controller jointly, rather than separately.

#### 1.4.1 人形机器人控制器

人形机器人，或者双足机器人，是复杂的非线性、强耦合的系统

#### 1.4.2 强化学习 /Reward shaping

#### 1.4.4 形态优化/联合优化



## 2. Controller

为了使机器人能够稳定、鲁棒的运动并完成目标任务，我们需要一个well-designed的控制器。在本设计中，我们所用到的机器人模型是一个人形机器人。人形机器人系统是一个复杂的非线性、强耦合的系统。在传统控制方法中，需要得到机器人的物理参数，如质量、密度、尺寸、转动惯量等，进而得到机器人的动力学方程，再通过各种算法实现对机器人的控制。

对于双足机器人和人形

2.1 