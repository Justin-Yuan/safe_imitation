from gym.envs.registration import register

register(
    id='SimpleNavigation-v0',
    entry_point='safe_il.envs:SimpleNavigation',
    max_episode_steps=1000,
    reward_threshold=200,
)
