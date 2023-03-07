'''
humanoidCustom.py
复制自 anaconda3\envs\GYM\Lib\site-packages\gym\envs\mujoco\humanoid_v3.py
在此基础上自定义mujoco ENV环境
'''

import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
import os


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 2.0)),
    "elevation": -20.0,
}


def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()


class HumanoidCustomEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        xml_file="E://CASIA//gym_RobotEvolution//gym_custom_env//assets//humanoid_custom.xml",
        forward_reward_weight=1.25,
        ctrl_cost_weight=0.1,
        contact_cost_weight=5e-7,
        contact_cost_range=(-np.inf, 10.0),
        healthy_reward=5.0,                     # 存活奖励
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.0),
        reset_noise_scale=1e-2,
        exclude_current_positions_from_observation=True,
    ):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._contact_cost_range = contact_cost_range
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        print("=========== HUMANOID Custom Env ============")
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    @property
    def healthy_reward(self):
        # 机器人正常运行的reward值，_healthy_reward默认为5，即正常训练时healthy_reward = 5
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        # 控制花费。所有控制量的开方和。
        control_cost = self._ctrl_cost_weight * np.sum(np.square(self.sim.data.ctrl))
        return control_cost

    @property
    def contact_cost(self):
        contact_forces = self.sim.data.cfrc_ext
        contact_cost = self._contact_cost_weight * np.sum(np.square(contact_forces))
        min_cost, max_cost = self._contact_cost_range
        contact_cost = np.clip(contact_cost, min_cost, max_cost)
        return contact_cost

    @property
    def is_healthy(self):
        # 机器人状态是否正常，通过qpos[2]的z轴位置判断（是否跌倒）
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.sim.data.qpos[2] < max_z

        return is_healthy

    @property
    def done(self):
        # episode是否结束的标志，在step()函数中返回
        # 如果机器人的状态是unhealthy，则done = True，训练终止
        done = (not self.is_healthy) if self._terminate_when_unhealthy else False
        return done

    def _get_obs(self):
        # obs空间
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        com_inertia = self.sim.data.cinert.flat.copy()
        com_velocity = self.sim.data.cvel.flat.copy()

        actuator_forces = self.sim.data.qfrc_actuator.flat.copy()
        external_contact_forces = self.sim.data.cfrc_ext.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        return np.concatenate(
            (
                position,
                velocity,
                com_inertia,
                com_velocity,
                actuator_forces,
                external_contact_forces,
            )
        )

    def print_obs_space(self):
        # 打印obs空间
        print('=========== q position ===========')
        position = self.sim.data.qpos.flat.copy()
        print('shape:',position.shape)
        print(position)

        print('=========== q velocity ===========')
        velocity = self.sim.data.qvel.flat.copy()
        print('shape:',velocity.shape)       
        print(velocity)

        print('=========== com_inertia ===========')
        com_inertia = self.sim.data.cinert.flat.copy()
        print('shape:',com_inertia.shape) 
        print(com_inertia)

        print('=========== com_velocity ===========')
        com_velocity = self.sim.data.cvel.flat.copy()
        print('shape:',com_velocity.shape) 
        print(com_velocity)

        print('=========== actuator_forces ===========')
        actuator_forces = self.sim.data.qfrc_actuator.flat.copy()
        print('shape:',actuator_forces.shape) 
        print(actuator_forces)

        print('=========== external_contact_forces ===========')
        external_contact_forces = self.sim.data.cfrc_ext.flat.copy()
        print('shape:',external_contact_forces.shape) 
        print(external_contact_forces)

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        return np.concatenate(
            (
                position,
                velocity,
                com_inertia,
                com_velocity,
                actuator_forces,
                external_contact_forces,
            )
        )    

    def step(self, action):
        '''
        step(self, action) -> observation, reward, done, info
        '''
        xy_position_before = mass_center(self.model, self.sim)
        self.do_simulation(action, self.frame_skip)
        xy_position_after = mass_center(self.model, self.sim)

        # dt为父类mujoco_env中的一个@property函数 
        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        # cost值 控制cost + 接触力cost
        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        # reward值 
        # _forward_reward_weight 默认为1.25
        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost + contact_cost

        observation = self._get_obs()
        reward = rewards - costs
        done = self.done
        info = {
            "reward_linvel": forward_reward,
            "reward_quadctrl": -ctrl_cost,
            "reward_alive": healthy_reward,
            "reward_impact": -contact_cost,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }

        return observation, reward, done, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
