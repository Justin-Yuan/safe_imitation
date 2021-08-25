"""
Simple navigation environment with obstacles to avoid to reach the goal.

To demo the environment and visualize the randomized scene try:

python safe_il/envs/simple_navigation.py
"""

import numpy as np

import gym
from gym import spaces
from gym import utils
from gym.utils import seeding

import logging
logger = logging.getLogger(__name__)

HORIZON = 100


class SimpleNavigation(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human']}

    def __init__(self) -> None:
        self.name = 'Simple Navigation'

        # Action space is two floats, [horizontal force, vertical force]
        # Both forces act within the range -1 to +1
        self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)

        self.observation_space = None

        self.viewer = None

        self.reset()

    def step(self, action):
        print('Step successful!')
        print(action)

        reward = 0
        done = False
        state = [0]
        return np.array(state, dtype=np.float32), reward, done, {}

    def reset(self):
        print('Environment reset')

    def render(self, mode='human'):
        print('Rendering Frame')

    def close(self):
        print('Closing')    

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)


def demo_simple_navigation(env, seed=None, render=False):
    env.seed(seed)
    total_reward = 0

    state = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        total_reward += reward

        if render:
            env.render()

        if done:
            break

    env.close()
    print(total_reward)


if __name__ == "__main__":
    demo_simple_navigation(SimpleNavigation(), render=False)
