#  Undergraduate Thesis

  

## 1.  Introduction

### 1.1 Background

Robots are often designed to work in complex terrains and environments. The robots that work in different environments requires the corresponding structure, including the actuators, the morphology, the size, etc. For example, a robot driving in rugged mountainous terrain needs more flexible joints to help it to cross obstacles and climb. Thus, good structural design is critical to a robot's ability to perform tasks in complex environments. On the other side, the manual design for robot structure is time consuming and requires necessary expertise.

Nowadays, there are more and more types and functions of robots, the bipedal robots and humanoid robots are developing rapidly. Compared with traditional wheeled and tracked robots, the design and control of bipedal robots are much more complex, requiring consideration of balance, gait and other issues. At the same time, the advantages of bipedal robots are also very prominent, as they are more adaptable, adapt to more diverse scenarios, have certain barrier-crossing capabilities, and can also perform multiple types of work.

But designing such a robot is a very long process, including mechanical structure design, morphological parameters design, electrical system design, controller design, and many other parts. Each of these parts is time-consuming. For example, the atlas robot in figure \ref{fig:a}, developed by Boston Dynamics, is a hydraulically driven, very powerful humanoid robot. Its first prototype was born in 1983 and has evolved over four decades into what you see today.

In recent years,  thanks to improvements in computing power and algorithms, artificial intelligence can be used to control and optimize robots. Many researchers are already using artificial intelligence as a tool for robotics research. Among them, the relatively mature development direction includes the combination of machine vision and robotic arm, reinforcement learning control of robotic arm to complete specific tasks, reinforcement learning control of quadruped robot, etc. Most of the work comes from simulation, and a small amount of work implements Sim-to-Real.

In addition, some other works combine artificial intelligence with the concept of agent in reinforcement learning, creating a new concept: embodied intelligence. Scholars study what kind of intelligence these agents with actual physical structure (in simulation) can show through neural network or reinforcement learning training. The concept of embodied intelligence leads to the research on the relationship among agent's perception, learning, control and physical structure. In fact, the physical structure of agent has a very strong relationship with its control and learning.

From the embodied intelligence in simulation to the actual robot, there is also intelligence in the morphology of robot. There is a saying: form determines function. Therefore, if you want to design a stable and robust robot that can complete specific functions, a good shape design is very necessary.

"Robot Evolution" is one of the concerned direction in the intersection of artificial intelligence and robot. Research in this field is devoted to using artificial intelligence to study robots, including but not limited to robot controller design, shape optimization, human-computer interaction, and so on.  Robot Evolution is an intelligent approach to robot optimization and design, and ultimately provides the optimal robot design for a specific task. This is a very valuable research direction. Although most of the current work is still carried out on the simulation platform, relevant work can guide the actual design of robots and serve as a reference method for robot design.

There are many interesting branches: for example, using a lot of computing resources to realize the evolution of an agent population in the simulation environment, observing the emergence of forms and actions in the population \cite{gupta2021embodied}. Or "complete" evolution of a single agent, growing limbs and learning to walk \cite{zhao2020robogrammar}, etc.

### 1.2 Motivation

Inspired by relative works, we will take a humanoid robot as agent to study the relationship between its form and control in this thesis. Through this thesis, we hope to come up with a joint optimization algorithm that can perform co-optimization of robot morphology and controller under specific tasks, to accelerate the design and optimization of humanoid robots, while using artificial intelligence methods to make robots more intelligent. Our algorithm will use reinforcement learning as the controller of the robot (agent), while using heuristic algorithms to optimize the robot morphology. In the process of learning and optimization, the robot will be able to perform better in the reinforcement learning environment and possibly produce different movements and postures.

 As mentioned above, the design of robot is a very complicated project. Robot morphology and controller are two crucial parts of the robot, they complement each other and cooperate with each other. Only appropriate morphology design with excellent performance of the controller, the robot can complete the target task. 

In the traditional robot development process, if the design structure is changed, the dynamics parameters of the robot will also change, which means that the controller will have to be modified. Therefore, the morphological structure of the robot is usually determined first, and then the controller is developed. Whether the proposed morphological structure is optimal for a particular task is generally judged by the experience of the engineer. Our algorithm will be an attempt to use a computer to jointly optimize the controller and morphology. Since the controller we use is based on reinforcement learning, it can be model free, which means we don't need to know any of the robot's dynamic parameters, as long as the robot model in the simulation is reasonable.

At the same time, our algorithm framework can be applied to a variety of robots. Given the right simulation platform and clear task, our proposed algorithm has the ability to optimize any robot. This will lay the groundwork for possible follow-up work. Some optimization tests and evaluations can also be carried out on the basis of our algorithm.

In addition, our thesis is also inspired by relevant researches in the field of "embodied intelligence". Some studies have shown that there is also potential intelligence in the morphology of agent. For example, to train a hand to grasp by reinforcement learning, a hand that looks like a human hand will definitely perform better than a hand that looks like a cat's paw. Thus, form optimization is also a way to generate intelligence. Our work will also be an attempt to optimize the form of a humanoid robot to see if it exhibits new capabilities for specific tasks.

### 1.2  Content



<img src="E:\CASIA\A-毕业设计\IMG\图片1.svg" alt="图片1" style="zoom: 33%;" />

We are going to propose an algorithm that can jointly optimize the controller and morphology of a humanoid robot,  which enable to get a better design in specific task. 

In order to complete the optimization of the robot morphology, several robots with different design parameters need to be evaluated.  The simplest way to make it is to give all the possible design parameters, evaluate each one, and select the one that performs best. However, the robot we are going to optimize is a humanoid robot with more than 20 links, which has a enormous design space. Besides, the evaluation process requires that each robot has an optimal controller. It would be very time-consuming and laborious to design the controllers for each robot with different morphology parameters individually. Therefore, how to quickly design a controller for each form has become the focus of research.

现有的相关研究中，机器人或智能体的形态优化算法通常分为两部分：外层的形态优化和内层的控制器优化。内层的控制器主要分为几种：

1. 传统控制方法，对系统进行动力学建模。这种方法是比较早期的方法，并不耗费计算资源，但是要求对系统的各个动力学参数已知，同时建模过程比较复杂，只适合于比较简单的系统。
2. 使用代理模型来进行控制。该方法的优点是不需要系统的具体动力学方程，但是控制效果较差，且往往需要其他的控制器作为辅助。
3. 使用MPC控制。这是目前比较主流的控制方法，同样需要系统的动力学模型，控制效果很好。
4. 使用神经网络控制。通常使用强化学习。该方法可以做到完全的model free，不需要花费时间在对系统建模上。但缺点是需要大量计算资源且耗费时间。

Since the humanoid robot model we selected has nearly 20 control parameters (17 joint drives), its interaction with the environment in the simulation will be a very complex system, so we chose reinforcement learning as the controller. In this way, we can complete the control of agent without knowing the specific dynamic equation of the system. At the same time, reinforcement learning itself is a process of controller optimization. Thus, the whole reinforcement learning process becomes our inner loop.

For morphology optimization, genetic algorithm (GA) will be selected as a heuristic search algorithm, and the object to be optimized is the length of limbs of humanoid robots. In GA, we will use the most basic way to encode morphological parameters. In order to reduce the complexity of the algorithm, we do not intend to optimize the torso, head and other parts. Since our goal for optimization is a humanoid robot, which itself is a sound design, we did not use the graph method to encode the morphology. The graph and graph search approach is more suitable for a robot with simple structure, or a robot that is developed from scratch. For a robot that already has a reasonable structure, all we need to do is to adjust the morphological parameters. If humanoid robots were to evolve to have three heads and six arms like Superman, the problem would become very complicated, especially there already have a reinforcement learning as the inner layer.

Figure \ref{fig:1-2} shows the overall research framework of this thesis. We built the simulation platform based on Mujoco and Gym, use pytorch to build the reinforcement learning neural network, use genetic algorithm as the outer layer optimization algorithm, and constructed the cooperation-morphology optimization framework by ourselves. 

Figure 1-3 shows the whole framework of the algorithm. The co-optimization divided into two parts: controller optimization and morphology optimization. The controller optimization will be the inner layer, using reinforcement learning; as well as the morphology optimization is the outer later, using heuristic search algorithm. The evaluation of the method will be done in a simulation environment. The reward that the agent get in the reinforcement learning environment will be used to evaluate the morphological design of the robot. We hope that through optimization at the morphological level, the robot can obtain higher rewards in the reinforcement learning environment.

The main content of the thesis comprises the two-layer co-optimization algorithms development and the simulation environment construction. The high-level optimization is the morphology optimization, and the low-level is the controller optimization. All experiments and verification of the algorithm will be conducted on the Mujoco simulation platform. The terrains include two types: flat floor, stairs. The prototype of the humanoid robot model used in the thesis was taken from the Mujoco simulation platform and packaged as an environment in gym, a platform commonly used in the field of reinforcement learning today.

### 1.3 Research background and significance



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

This was confirmed by the work of Cheney et al. who demonstrated that controller optimization of a soft robot can contribute to its morphological optimization. This work is of high value for similar joint optimization problems: it points out that optimization should be hierarchical and synergistic, rather than a simple superposition of two optimizations. This problem requires better coordination between controller optimization and design optimization to improve the efficiency of collaborative optimization.

In the early research, the main optimization method was evolutionary algorithm, that is, to optimize the whole population through evolutionary methods, and search in the population. The advantage of evolutionary algorithm is that it can deal with discrete design space, such as different kinds of joints of the robot, different kinds of links and so on. However, it is also obvious that the evolutionary algorithm is prone to local optimization and needs to evaluate every individual in the population, which is very time consuming and low sample efficiency.

#### 1.4.2 Recently Reserch

Then came the era of rapid development of GPU and artificial intelligence. Methods including evolutionary optimization and reinforcement learning have made great achievements in co-optimization of morphology and control, and have been extended to include different robot forms such as manipulators and soft bodies.  

With more advanced equipment paired with a better simulation environment, the performance of some joint optimization algorithms is further unlocked. One very large branch was born: the use of graphs to represent robotic forms and to optimize their structure. Wang et al. used the graph to represent the structure of the robot, and each connecting rod of the robot was set as the node of the graph. Thus, the robot form optimization problem is transformed into a graph search problem. At the same time, the concept of population is still added, and evolutionary search of graph is realized by mutation and mutation operator which can add and delete nodes. The linkage configuration of the robot itself is very suitable to be defined by graph, which greatly improves the efficiency of search.

Similarly, Zhao et al. also use a graph to represent a robot. They used a new generative grammar, "RoboGrammar," to generate a variety of robots. Heuristic graph search is used to optimize the resulting structure. It is worth mentioning that most of the robots generated in their work are relatively complex. In order to realize the control of the newly generated robots, they use MPC as the controller. In this work, the MPC is stable, robust and can adapt to different robot structures. The result of their work is very impressive, a very interesting and efficient way of generating robots from syntax, a combination of syntax and heuristic search, to generate many different shapes of robots. Scaff et al. also adopted a similar method and proposed the "N-Limb" algorithm, Neural Limb Optimization. They combined with neural networks to further improve the optimization performance of the robot design-control pair in a large design space.

The method of using graph to construct robot has been proved to be effective by many works. Because of the structural characteristics of the robot, the graph is very suitable to represent the structural optimization and development of the robot. Such as adding limbs to the torso and lengthening the limbs by adding joints to the limbs. With the improvement of robot complexity, the requirement of controller is higher and higher. It is impossible to evaluate the performance of a newly generated structure without a stable controller.

In addition to using graphs to represent robots, there are many jobs that use graph neural networks as controllers. This largely solves Chenery's problem of inadequate controller optimization during evolution. The biggest benefit of using graph neural networks as controllers is the controller's ability to migrate between different designs. Wang's work \cite{wang2018nervenet} mentioned above constructed a graph neural network and named it NerveNet.

Zhang et al. used an agent model to model a bipedal robot. He proposed a computer simulation-based co-evolutionary strategy for the structure and and gait of the bipedal robot, using 3DLIPM as the controller and using the agent model for computing, thus reducing the computational cost of the robot controller. The agent model is a data-driven model that can be ignored for parameters within the system. This approach simplifies the controller of the robot and is difficult to use for complex control tasks. Similarly, Ha et al. used implicit functions to model the dynamics equations of a robotic system. The optimization of the implicit function is used to optimize the design parameters and control parameters of the robot.

These works have used a variety of approaches for robot control strategies, including modeling system dynamics and performing conventional control, using MPC control, and using 3DLIPM control. The use of graphs to encode robot designs greatly improves the efficiency of updating design parameters during the evolution of the robot, and as a result, the development of control strategies will largely limit the upper limit of robot evolution algorithms. Traditional control methods all require knowledge of the system dynamics model and take a large amount of time to design. These methods perform well for simple robots, but when the robot and the task become complex, it becomes infeasible to design a controller for each form. Although the use of "black box models" or implicit functions has been proposed to simplify complex systems, this still limits the effectiveness of controllers. The advent of neural networks and reinforcement learning will go a long way to solve this problem.

#### 1.4.3 Neural Network Approaches

With the development of neural networks, more and more research is devoted to combining neural networks and robotics. This includes using neural networks as controllers for robots, or using neural networks for joint optimization of robots. The joint optimization of robot morphology and controller based on neural networks is the research of this paper, but these are based on the research of neural networks as robot controllers.

The development of robot control using neural networks does not have a long history. Compared to other classical applications of neural networks, CV, NLP, robot control does not have as much data to learn. Therefore, the development of this direction mainly lies in the use of reinforcement learning to accomplish the control of robots on simulation platforms. 

The field was boosted by the birth of DQN networks, a reinforcement learning network that was proposed by Mnih et al. in 2015 and published in Nature. Early reinforcement learning could only accept discrete, low-dimensional observation spaces as input to the intelligences. This is very much at odds with real-world situations, so early reinforcement learning can only be used for simple tasks such as Atari games. DQNs, on the other hand, can derive valid information from high-dimensional sensory inputs and generalize past successes to new situations. This work successfully connects high-dimensional input and action output, resulting in the first agent capable of learning to perform well in a variety of challenging tasks. This is a milestone work, marking a new stage in reinforcement learning.

Heess et al. applied reinforcement learning to a relatively complex humanoid robot model. They designed a variety of complex terrains and provided simple reward functions for humanoid robots to perform reinforcement learning training as agents. The results showed that although no expert knowledge was given to the agent, only its own state and surrounding terrain information were used as the observation space for the agent, the agent still generated a variety of complex motor skills and gaits during training. This shows that reinforcement learning as a controller has a very good evolutionary effect, and intelligence will be generated automatically during the learning process. Most importantly, reinforcement learning is model free. It does not require any specific dynamics parameters of the robot, and only requires a reasonable design of agent observation space and reward function. That is, of course, if you have a sufficiently powerful simulation environment.

Another classic work using reinforcement learning as a control strategy for robots, Peng et al. proposed the 'DeepMimic' framework, which is a framework similar to imitation learning. Given a series of actions, the robot is allowed to learn through reinforcement learning. DeepMimic builds neural networks and adds the degree of similarity to the target action in the reward function, thus facilitating agent learning. Eventually, the robot is able to learn very complex movements such as spinning kicks and flips. Also, he proposed multi-task reinforcement learning, based on which it is possible to make a robot walk and flip at the same time, or even specify a random point and let the robot kick to that point by spinning kick. This work is applied to computer animation design.

The field of robot evolution has been developed once again based on the use of neural networks as robot controllers. Generating new morphs is not difficult, as long as the morphs are specifically and carefully coded, a variety of new designs can be generated. The difficulty lies in how to evaluate the morphs, which requires designing a good controller for each morph. The emergence of neural networks provides a new way of thinking about this problem. In fact, the graph neural network mentioned in the previous section is one of the methods. A more common approach is reinforcement learning.

Ha proposed a method to update the agent design by reinforcement learning. He sets the design parameters of the agent as parameters that can be learned. As a result, in the process of reinforcement learning, not only the control strategy will be learned, but also the structure of itself will be updated. This is a relatively simple approach, not layering morphological and control designs, and is a preliminary experimental work. Luck et al. estimate the performance of each morphological design by learning the Q function and update the morphological design during the iterative process of reinforcement learning. He maintains two sets of networks simultaneously, the global control network ($\pi_{Pop.}, Q_{Pop.}$, which is actually the Actor network and the Critic network) and the individual control network ($\pi_{Ind.}, Q_{Ind.}$) for this design. This approach is able to reduce the training time of the individual control network and improve the overall optimization efficiency.

A joint optimization framework based on reinforcement learning was proposed by Scaff et al. Still using reinforcement learning for learning and optimizing the control strategy of an intelligent body, Gaussian mixture distribution is used to update the morphological parameters in the morphological design. The two are jointly optimized by the PPO algorithm. However, the learning convergence becomes difficult because the design space is too large and the search of morphology and controller is difficult to decouple. Scaff's proposed joint optimization framework can be adapted to more complex robots. The work in this paper is largely inspired by him and tries to make some innovations.

Gupta et al. developed an environment called "evolutionary playground" and a computational framework called "Deep Evolutionary Reinforcement Learning" (DERL) to explore the relationship between embodied intelligence and environment. In this "evolutionary playground", a large population is maintained, in which each agent learns reinforcement and iterates through unique genetic mechanisms. This paper also verifies the Baldwin effect in evolutionary biology. In this work, a concept called "embodied intelligence" is involved. For more information on embodied intelligence, see this review by Liu et al., which discusses the importance of the form of embodied intelligence and the relationship between form and intelligence. Most of the work mentioned above is carried out in the simulation environment, in which the use of reinforcement learning training can become embodied intelligence. Embodied intelligence is defined as an agent with physical properties and actual structure. Work has shown that form also involves intelligence. Evolutionary form is also a way for agents to develop their intelligence. These views are very instructive to our work.







## 2. Controller

In order to enable the robot to move stably and robustly and complete the target task, we need a well-designed controller.  As mentioned above, the controller is the key point in the robot evolution process. The effect of the controller and the cost of iteration determine the upper limit of the joint optimization algorithm. In this chapter, we will discuss controller design methods for humanoid robots.

### 2.1 仿真环境

We need a simulation environment where humanoid robots can be used and various terrain can be added. The choice of simulation environment is very important. It can provide an effective method to evaluate and improve the performance and behavior of robot systems. Through the simulation environment, various robot application scenarios can be simulated, so as to better predict the performance of the robot in the actual environment, and optimize the performance of the robot system.

When selecting the simulation environment, the reliability and accuracy of the simulation environment should be taken into account. The behavior and performance of the robot system must be accurately simulated to ensure the reliability and accuracy of the research results. At the same time, considering the flexibility and extensibility of the simulation environment, it should be able to support different robot systems and scenarios. Most importantly, the development cost of the simulation environment should not be too high. If the simulation environment is too professional and difficult to develop, it is not suitable for us.

All things considered, we chose Mujoco as the simulation environment.

Mujoco is a high-quality multi-rigid body simulation engine that can be used to simulate various types of physical systems. Most importantly, it is open source software that supports C++, Python, and other programming languages, and it is mature. Mujoco uses an event-driven modeling approach to define the motion and interaction behavior of the model. It supports a variety of physical models, including rigid body motion, fluid dynamics, electromagnetic fields, etc., and can be easily integrated with other simulation software and libraries. Mujoco also provides a wealth of visual tools to help users observe and debug simulation systems. Features such as motion capture, virtual reality, graphics rendering and animation are included to make it easy to create and display simulation results.



### 2.2 RL

We usually model reinforcement learning as a Markov decision process (MDP). The MDP consist of a tuple $\langle \mathcal{S}, \mathcal{A},P,\mathcal{R},\mathcal{\gamma} \rangle$, among the tuple:

- $\mathcal{S}$  : The state space, or the observation space.代表状态空间
- $\mathcal{A}$ : The action space. 
- $P$  : $P(s′|s,a) $ is the state transition function, represents the probability that the state changes to $s' $ after the action $a$  is taken in the $s$ state. 
- $R$ : The reward function.
- $\gamma$: The discount factor, which is a constant, $\gamma \in [0,1] $.

$ R_t = \sum^{T}_{t'=t}{\gamma^{t'-t}r_{t'}}$

智能体的**策略**（Policy）通常用字母$表示。策略是一个函数，表示在输入状态情况下采取动作的概率。当一个策略是**确定性策略**（deterministic policy）时，它在每个状态时只输出一个确定性的动作，即只有该动作的概率为 1，其他动作的概率为 0；当一个策略是**随机性策略**（stochastic policy）时，它在每个状态时输出的是关于动作的概率分布，然后根据该分布进行采样就可以得到一个动作。在 MDP 中，由于马尔可夫性质的存在，策略只需要与当前状态有关，不需要考虑历史状态。回顾一下在 MRP 中的价值函数，在 MDP 中也同样可以定义类似的价值函数。但此时的价值函数与策略有关，这意为着对于两个不同的策略来说，它们在同一个状态下的价值也很可能是不同的。这很好理解，因为不同的策略会采取不同的动作，从而之后会遇到不同的状态，以及获得不同的奖励，所以它们的累积奖励的期望也就不同，即状态价值不同



Another important element is policy. Policy is a kind of mapping, write for $\pi: S \rightarrow Δ(A)$, where $Δ (A) $ represents the action space on probability distribution. In this case, we denote $P (a_t = a | s_t = s) $  as $\pi(a|s) $, which represent the probability of take an action $a$ under the state $s$.

If the strategy is deterministic, then we can write the strategy as $π:S→A$. In this case, we can write the action resulting from executing the strategy $π$ in the state $s$ as $π(s)$. Strictly speaking, the above definition is only for the definition of "stability strategy" , that is the distribution of action or action produced by the policy is not affected by time.



$V^{\pi}(s)$ is used to represent the state value function of following the policy $\pi$ in MDP, which is defined as the expected return that can be obtained from following the policy $\pi$ from the state $s$. The mathematical expression is as follows:
$$
V^\pi(s) = \mathbb{E}_\pi[G_t|S_t=s]
$$
In MDP, an additional action value function is defined because of the existence of the action. Use $Q^\pi (s,a)$ to represent the expected return that can be obtained by taking action $a$ under the current state $s$ when following the policy$\pi$. The mathematical expression is as follows:
$$
Q^\pi(s,a) = \mathbb{E}_\pi[G_t|S_t=s,A_t=a]
$$
Thus, the relation between the state value function and the action value function can be obtained: when following the policy $\pi$, the value of the state $s$ is equal to the result of multiplying the probability of all the actions that can be taken based on the policy$\pi$ in this state by the action value.
$$
V^\pi(s) = \sum_{a \in A}{\pi{(a|s)}Q^\pi(s,a)}
$$
When following the policy $\pi$, the value of the action $a$ taken under the current state $s$ is equal to the immediate return plus the decay of all possible state transition probabilities of the next state multiplied by the corresponding value:
$$
Q^\pi(s,a) = r(s,a)+ \gamma \sum_{s' \in S}{P(s'|s,a)V^\pi(s')}
$$

$$
J(\theta) =  \mathbb{E}_{s_0}[V^{\pi_\theta}(s_0)]
$$

$$
\nabla_\theta J(\theta) =  \mathbb{E}_{\pi_\theta}[Q^{\pi_\theta}(s,a)\nabla_\theta\log\pi_\theta(a|s)]
$$

$$
\nabla_\theta J(\theta) =  \mathbb{E}_{\pi_\theta}[\sum_{t=0}^T\psi_t\nabla_\theta\log\pi_\theta(a_t|s_t)]
$$

$$
\mathcal{L}(\omega) = \frac{1}{2}(r+\gamma V_\omega(s_{t+1})- V_\omega(s_t))^2
$$

$$
\nabla_\omega\mathcal{L}(\omega) = -(r+\gamma V_\omega(s_{t+1})- V_\omega(s_t))\nabla_\omega V_\omega(s_t)
$$

$$
\pi^* = \mathop{\arg\min}\limits_{\pi}\mathbb{E}_\pi [\sum_tr(s_t,a_t)+\alpha H(\pi(\cdot|s_t))]
$$

$$
L_Q(\omega)= \mathbb{E}_{(s_t,a_t,r_t,s_{t+1})\sim R,a_{t+1}\sim\pi_\theta(\cdot|s_{t+1})}[\frac{1}{2}\Big(Q_\omega(s_t,a_t) \\
-\big(r_t+\gamma(\mathop{\min}\limits_{j=1,2}Q_{\omega_j^-}(s_{t+1},a_{t+1})-\alpha\log\pi(a_{t+1}|s_{t+1}))\big)\Big)^2]
$$



The loss function of the policy is given by the KL divergence. For the environment of continuous action space, the strategy of SAC algorithm outputs the mean and standard deviation of the Gaussian distribution, but the process of sampling the action according to the Gaussian distribution is not derivable. Therefore, we need to use the reparameterization trick. The loss function of policy $\pi$ can be expressed as:
$$
L_\pi(\theta)=\mathbb{E}_{s_t\sim R,\epsilon_t\sim \mathcal{N}}\Big[\alpha\log\big(\pi_\theta(f_\theta(\epsilon_t;s_t)|s_t)\big)- \mathop{\min}\limits_{j=1,2}Q_{\omega_j}(s_t,f_\theta(\epsilon_t;s_t))\Big]
$$
$\omega_1^- \leftarrow \omega_1$, $\omega_2^- \leftarrow \omega_2$    $Q_{\omega_1^-}$

$y_i = r_i + \gamma \min_{j=1,2}Q_{\omega_j^-}(s_{i+1},a_{i+1})-\alpha\log\pi_\theta(a_{i+1}|s_{i+1})$
$$
L_Q = \frac{1}{N}\sum_{i=1}^{N}(y_i-Q_{\omega_j}(s_i,a_i)^2) \quad \quad \quad j=1,2
$$
$\widetilde{a}_i$
$$
L_\pi(\theta)=\frac{1}{N}\sum^{N}_{i=1}\big(\alpha \log \pi_\theta(\widetilde{a}_i|s) - \mathop{\min}\limits_{j=1,2} Q_{\omega_j}(s_i,\widetilde{a}_i) \big)
$$

$$
\omega_1^- \leftarrow \tau\omega_1 + (1-\tau)\omega_1^- , \quad \omega_2^- \leftarrow \tau\omega_2 + (1-\tau)\omega_2^- 
$$

$ [-\infin,+\infin]$


$$
R = w_{forward}r_{foward} + w_{healthy}r_{healthy} + w_{stand}r_{stand} - w_{control}c_{control} - w_{contact}c_{contact}
$$

$$
c_{con} = \sum_{i=1}^{N}{F_{ext}}^2
$$

$$
c_{ctrl} = \sum_{i=1}^{N}{\tau_{input}}^2
$$

$$
r_{p} = (\vec{z}_{world}\cdot\vec{z_{agent}} + 1)/2 + (0.5-{y_{center}}^2)
$$

