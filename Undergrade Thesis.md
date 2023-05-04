#  Undergraduate Thesis

  

## 1.  Introduction

### 1.1 Introduction


In recent years,  thanks to improvements in computing power and algorithms, artificial intelligence can be used to control and optimize robots. Many researchers are already using artificial intelligence as a tool for robotics research. Among them, the relatively mature development direction includes the combination of machine vision and robotic arm, reinforcement learning control of robotic arm to complete specific tasks, reinforcement learning control of quadruped robot, etc. Most of the work comes from simulation, and a small amount of work implements Sim-to-Real.

In addition, some other works combine artificial intelligence with the concept of agent in reinforcement learning, creating a new concept: embodied intelligence. Scholars study what kind of intelligence these agents with actual physical structure (in simulation) can show through neural network or reinforcement learning training. The concept of embodied intelligence leads to the research on the relationship among agent's perception, learning, control and physical structure. In fact, the physical structure of agent has a very strong relationship with its control and learning.

From the embodied intelligence in simulation to the actual robot, there is also intelligence in the morphology of robot. There is a saying: form determines function. Therefore, if you want to design a stable and robust robot that can complete specific functions, a good shape design is very necessary.

“Robot Evolution” is one of the concerned direction in the intersection of artificial intelligence and robot. Research in this field is devoted to using artificial intelligence to study robots, including but not limited to robot controller design, shape optimization, human-computer interaction, and so on. There are many interesting branches: for example, using a lot of computing resources to realize the evolution of an agent population in the simulation environment, observing the emergence of forms and actions in the population. Or "complete" evolution of a single agent, growing limbs and learning to walk, etc.

Inspired by relative works, we will take a humanoid robot as agent to study the relationship between its form and control in this thesis. We propose a joint optimization algorithm based on genetic algorithm and reinforcement learning, which can jointly optimize the controller and shape of humanoid robot and make it have better performance. Our algorithm will use reinforcement learning as the controller of the robot (agent), while using heuristic algorithms to optimize the robot morphology. In the process of learning and optimization, the robot will be able to perform better in the reinforcement learning environment and possibly produce different movements and postures.

### 1.2  Content

We are going to propose an algorithm that can jointly optimize the controller and morphology of a humanoid robot,  which enable to get a better design in specific task. 

In order to complete the optimization of the robot morphology, several robots with different design parameters need to be evaluated.  The simplest way to make it is to give all the possible design parameters, evaluate each one, and select the one that performs best. However, the robot we are going to optimize is a humanoid robot with more than 20 links, which has a enormous design space. Besides, the evaluation process requires that each robot has an optimal controller. It would be very time-consuming and laborious to design the controllers for each robot with different morphology parameters individually.

 Therefore, a co-optimization strategy of morphology and controller is developed. The controller is optimized at the same time as the morphology is optimized. In this way, the total time cost can be significantly reduced.

<img src="E:\CASIA\A-毕业设计\IMG\图片1.svg" alt="图片1" style="zoom: 33%;" />

The co-optimization can be divided into two parts: controller optimization and morphology optimization. The controller optimization will be the inner layer, using reinforcement learning; as well as the morphology optimization is the outer later, using heuristic search algorithm. The evaluation of the method will be done in a simulation environment.

The reward that the robot can get in the reinforcement learning environment will be used as an evaluation index to judge the merits of the morphological design of the robot. We hope that through optimization at the morphological level, the robot can obtain higher rewards in the reinforcement learning environment.



Figure 2 shows the overall research framework of this thesis. We built the simulation platform based on Mujoco and Gym, use pytorch to build the reinforcement learning neural network, use genetic algorithm as the outer layer optimization algorithm, and constructed the cooperation-morphology optimization framework by ourselves. The details are explained later. 

The main content of the thesis comprises the two-layer co-optimization algorithms development and the simulation environment construction. The high-level optimization is the morphology optimization, and the low-level is the controller optimization. All experiments and verification of the algorithm will be conducted on the Mujoco simulation platform. The terrains include two types: flat floor, stairs. The prototype of the humanoid robot model used in the thesis was taken from the Mujoco simulation platform and packaged as an environment in *gym*, a platform commonly used in the field of reinforcement learning today.

### 1.3 Research background and significance

Robots are often required to work in complex terrains and environments. The robots that work in different environments requires the corresponding structure, including the actuators, the morphology, the size, etc. For example, a robot driving in rugged mountainous terrain needs more flexible joints to help it to cross obstacles and climb. Thus, good structural design is critical to a robot's ability to perform tasks in complex environments. On the other side, the manual design for robot structure is time consuming and requires necessary expertise.

Nowadays, there are more and more types and functions of robots, and bipedal robots and humanoid robots are developing rapidly. Compared with traditional wheeled and tracked robots, the design and control of bipedal robots are much more complex, requiring consideration of balance, gait and other issues. At the same time, the advantages of bipedal robots are also very prominent, as they are more adaptable, adapt to more diverse scenarios, have certain barrier-crossing capabilities, and can also perform multiple types of work.

But designing such a robot is a very long process, including mechanical structure design, morphological parameters design, electrical system design, controller design, and many other parts. Each of these parts is time-consuming. For example, the atlas robot in Figure 1, developed by Boston Dynamics, is a hydraulically driven, very powerful humanoid robot. Its first prototype was born in 1983 and has evolved over four decades into what you see today.

Through this thesis, we hope to come up with a joint optimization algorithm that can perform co-optimization of robot morphology and controller under specific tasks, to accelerate the design and optimization of humanoid robots, while using artificial intelligence methods to make robots more intelligent. As mentioned above, the design of robot is a very complicated project. Robot morphology and controller are two crucial parts of the robot, they complement each other and cooperate with each other. Only appropriate morphology design with excellent performance of the controller, the robot can complete the target task. 

In the traditional robot development process, if the design structure is changed, the dynamics parameters of the robot will also change, which means that the controller will have to be modified. Therefore, the morphological structure of the robot is usually determined first, and then the controller is developed. Whether the proposed morphological structure is optimal for a particular task is generally judged by the experience of the engineer. Our algorithm will be an attempt to use a computer to jointly optimize the controller and morphology. Since the controller we use is based on reinforcement learning, it can be model free, which means we don't need to know any of the robot's dynamic parameters, as long as the robot model in the simulation is reasonable.

At the same time, our algorithm framework can be applied to a variety of robots. Given the right simulation platform and clear task, our proposed algorithm has the ability to optimize any robot. This will lay the groundwork for possible follow-up work. Some optimization tests and evaluations can also be carried out on the basis of our algorithm.

In addition, our thesis is also inspired by relevant researches in the field of "embodied intelligence". Some studies have shown that there is also potential intelligence in the morphology of agent. For example, to train a hand to grasp by reinforcement learning, a hand that looks like a human hand will definitely perform better than a hand that looks like a cat's paw. Thus, form optimization is also a way to generate intelligence. Our work will also be an attempt to optimize the form of a humanoid robot to see if it exhibits new capabilities for specific tasks.





### 1.4 Relevant research status

This section will give a brief introduce of the relative methods and approaches in the field of robot evolution and robot co-optimization in the recent years. 

#### 1.4.1 Co-optimization of robot 

The early beginning in this field is the seminal work done by Karl Sims in 1994, using an evolutionary method to optimize the morphology. The method is more like the Genetic Algorithm, they represent the genotype by directed graph, maintain a population $\{x_i\}_j$consist of several individuals. The genotype of each individual contains information of an agent, include the design parameters,  position of sensors, and the controller. Every individual will be evaluate in each generation by a fitness function $f(x)$. Between two generations, some biological method, such as mutation and crossover, will be used to the genotype. Sims found that during the evolution process, the agent can get intelligence and the complex movement will emerge to solve the specific task, as well as the morphology become more complex.  Without computing support at the time, Sims worked on very abstract creatures, but he creatively connected the evolutionary approach to agent.

![image-20230504184811472](C:\Users\孟一凌\AppData\Roaming\Typora\typora-user-images\image-20230504184811472.png)

Sims' work gives a direction of robot evolution, in the following two decades, this approach get a rapidly evolve. Since this work can help robots search for morphologies that better match to their environments and tasks, it got widely valued rapidly. The article published in Nature by Lipson et al. is very noteworthy. They used variable length cylindrical parts to build a real robot that could evolve. Using walking ability as an evaluation metric, it was iterated about 600 times in a simulation environment to complete the evolution, and commercial rapid prototyping techniques were used to translate it into a real physical system.

A lot of work after this has improved this framework. For example, the work of Hornby et al. improved the evolutionary algorithm by proposing a fully automated design system that generates robots based on regularity, hierarchy, reuse, and other rules. Also in another work, Hornby proposed a new way of coding agents using the "L-system".

Most of these earlier works focus on improvements to evolutionary algorithms. Basically having a unified framework for encoding a certain agent or robot, followed by using evolutionary algorithms to optimize this encoding to obtain a better design. However, one of potential problem is that in the above mentioned work, their use of evolutionary algorithms generally causes the morphological design of the agent and the controller to mutate once in each iteration of the population. This could lead to controllers and morphological designs that do not match, thus causing the results to fall into local optima. Due to the limited arithmetic power at that time, the agent structures used in the simulation were relatively simple, so the problem of controller and morphological adaptation was not prominent. If it is a very complex robot, then this problem will become serious. Although researchers at the time began to divide optimization into an inner layer of controller optimization and an outer layer of morphology optimization, the boundary between these two layers did not seem to be clear.

This was confirmed by the work of Cheney et al. who demonstrated that controller optimization of a soft robot can contribute to its morphological optimization. This work is of high value for similar joint optimization problems: it points out that optimization should be hierarchical and synergistic, rather than a simple superposition of two optimizations.

Then came the era of rapid development of GPU and artificial intelligence. Methods including evolutionary optimization and reinforcement learning have made great achievements in co-optimization of morphology and control, and have been extended to include different robot forms such as manipulators and soft bodies.  

With more advanced equipment paired with a better simulation environment, the performance of some joint optimization algorithms is further unlocked. One very large branch was born: the use of graphs to represent robotic forms and to optimize their structure. Wang等人使用图来表示



## 2. Controller

为了使机器人能够稳定、鲁棒的运动并完成目标任务，我们需要一个well-designed的控制器。在本设计中，我们所用到的机器人模型是一个人形机器人。人形机器人系统是一个复杂的非线性、强耦合的系统。在传统控制方法中，需要得到机器人的物理参数，如质量、密度、尺寸、转动惯量等，进而得到机器人的动力学方程，再通过各种算法实现对机器人的控制。

对于双足机器人和人形

2.1 