import gym
envs = gym.vector.make("CartPole-v1", num_envs=2, asynchronous=False)
envs = gym.wrappers.RecordVideo(envs, "videos", video_length=200)
envs = gym.wrappers.RecordEpisodeStatistics(envs)
envs.reset()
for i in range(1000):
    _, _, _, infos = envs.step(envs.action_space.sample())
    for info in infos:
        if "episode" in info.keys():
            print(f"i, episode_reward={info['episode']['r']}")
            break