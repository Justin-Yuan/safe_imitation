from gym.envs.registration import register

register(
    id='SimpleNavigation-v0',
    entry_point='safe_il.envs:SimpleNavigation',
    max_episode_steps=10000,
    reward_threshold=200,
)

register(
    id='BoneDrilling2D-v0',
    entry_point='safe_il.envs:BoneDrilling2D',
    max_episode_steps=10000,
    reward_threshold=200,
)
