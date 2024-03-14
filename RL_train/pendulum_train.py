'''
python pendulum_train.py 
'''
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="cfg", config_name="InvertedPendulumCfg.yaml")
def main(cfg : DictConfig) -> None:
    # ------- 来自于mujoco150在win+py3.9下的矫情的要求 --------
    # 手动添加mujoco路径
    import os
    from getpass import getuser
    user_id = getuser()
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    import gym_env       # 注册自定义环境
    import time
    import gym
    import torch
    import numpy as np
    from datetime import datetime
    from stable_baselines3 import SAC, TD3, PPO
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.env_util import make_vec_env
    
    print(OmegaConf.to_yaml(cfg)) # 打印配置

    # 随机种子
    seed = 1

    # 环境名
    env_id = cfg.env.env_id
    n_timesteps = cfg.train.n_timesteps
    model_name = cfg.env.model_name  #41 表示4 0.4 1 0.1
    algo = 'ppo'

    # dump config dict
    experiment_dir = os.path.join('runs', model_name + 
    '_{date:%d-%H-%M-%S}'.format(date=datetime.now()))
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
        
    # 存放在sb3model/文件夹下
    save_path = experiment_dir + f"/{model_name}{algo}_{env_id}"

    # tensorboard log 路径
    tensorboard_log_path = experiment_dir+'/tensorboard_log'
    tensorboard_log_name = f"{model_name}{algo}_{env_id}"


    env_kwargs = { "w_e":cfg.env.w_e,
                   "w_q1":cfg.env.w_q1,
                   "w_q2":cfg.env.w_q2,
                   "w_r":cfg.env.w_r,
                   "w_t":cfg.env.w_t,
                   "w_c":cfg.env.w_c,
                   "energy_obs":cfg.env.energy_obs}
                   
    # Instantiate and wrap the environment
    env = make_vec_env(env_id = env_id, n_envs = 15,env_kwargs = env_kwargs)


    # Create the evaluation environment and callbacks
    eval_env = Monitor(gym.make(env_id, **env_kwargs))

    callbacks = [EvalCallback(eval_env, best_model_save_path=save_path)]

    RLalgo = {
        "sac": SAC,
        "td3": TD3,
        "ppo": PPO,
    }[algo]

    hyperparams = dict(
            batch_size=cfg.train.batch_size,
            learning_rate=cfg.train.learning_rate,
            gamma=cfg.train.gamma,
            device=cfg.train.device
        )



    begin_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    model = RLalgo("MlpPolicy", env, verbose=1, tensorboard_log = tensorboard_log_path, **hyperparams,seed = seed)
    try:
        model.learn(n_timesteps, callback=callbacks , tb_log_name = tensorboard_log_name )
    except KeyboardInterrupt:
        pass
    print('=====================================')
    print(f"Saving to {save_path}.zip")
    model.save(save_path)
    end_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    print('Started at: ' + begin_time)
    print('Ended at: ' + end_time)
    print('=====================================')


if __name__ == '__main__':
    main()