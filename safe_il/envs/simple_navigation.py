"""
Simple navigation environment with obstacles to avoid to reach the goal.

To demo the environment and visualize the randomized scene try:

python safe_il/envs/simple_navigation.py
"""

from Box2D.Box2D import b2CircleShape
import numpy as np

import Box2D
from Box2D.b2 import (
    edgeShape,
    circleShape,
    fixtureDef,
    polygonShape,
    revoluteJointDef,
    contactListener,
)

import gym
from gym import spaces
from gym.utils import seeding, EzPickle


import logging
logger = logging.getLogger(__name__)

STATE_H = 64
STATE_W = 64
VIEWPORT_W = 512
VIEWPORT_H = 512
FPS = 50
SCALE = 30.0


class SimpleNavigation(gym.Env, EzPickle):
    metadata = {"render.modes": ["human", "rgb_array"],
                "video.frames_per_second": FPS}

    def __init__(self,
                 pixel_obs=False):
        self.name = 'Simple Navigation'

        # Action space is two floats, [horizontal force, vertical force]
        # Both forces act within the range -1 to +1
        self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)

        if pixel_obs:
            self.observation_space = spaces.Box(
                0, 255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8)
        else:
            # Observation space is distance in X and Y to Goal and Obstacles
            self.observation_space = spaces.Box(
                -np.inf, np.inf, shape=(8,), dtype=np.float32)

        self.viewer = None

        self.world = Box2D.b2World(gravity=(0, 0), doSleep=True)
        self.obstacles = []
        self.goal_pos = (5, 5)
        self.agent = None
        self.state = None
        self.done = None
        self.episode_steps = 0
        self.horizon = 500

        self.reset()

    def step(self, action):
        action = np.clip(action, -1, +1).astype(np.float32).tolist()
        # Apply force to the agent
        self.agent.ApplyForceToCenter(force=(action[0], action[1]), wake=True)

        # Box2D World Simulation
        timeStep = 1.0 / 60
        vel_iters, pos_iters = 6, 2
        self.world.Step(timeStep, vel_iters, pos_iters)

        reward = 0
        self.done = self.episode_steps >= self.horizon
        state = [0]
        return np.array(state, dtype=np.float32), reward, self.done, {}

    def reset(self, random_start=False):
        if random_start:
            agent_start_pos = self.np_random.rand(2)
        else:
            agent_start_pos = (2, 2)

        self.agent = self.world.CreateDynamicBody(
            shapes=b2CircleShape(pos=agent_start_pos, radius=0.5)
        )

        self.agent.color = (1, 0, 0)  # Red

        self.goal = self.world.CreateStaticBody(
            shapes=b2CircleShape(pos=self.goal_pos, radius=0.5)
        )
        self.goal.color = (0, 1, 0)  # Green

        self.drawlist = [self.agent, self.goal] + self.obstacles

        print('Environment has been reset!')

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W/SCALE, 0, VIEWPORT_H/SCALE)

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans * f.shape.pos)
                    self.viewer.draw_circle(
                        f.shape.radius, 20, color=obj.color
                    ).add_attr(t)
                else:
                    print('Not a circle')

        return self.viewer.render(return_rgb_array=(mode == "rgb_aray"))

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def _destroy(self):
        # Destroy all Box2D related entities to manage the memory efficiently
        self.world.DestroyBody(self.agent)
        self.world.DestroyBody(self.goal)
        for ob in self.obstacles:
            self.world.DestroyBody(ob)


def demo_simple_navigation(env, seed=None, render=False):
    env.seed(seed)
    total_reward = 0

    state = env.reset()
    for _ in range(500):
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
    demo_simple_navigation(SimpleNavigation(), render=True)
