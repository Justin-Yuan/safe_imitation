"""
Simple navigation environment with obstacles to avoid to reach the goal.

To demo the environment and visualize the randomized scene try:

python safe_il/envs/bone_drilling_2d.py
"""

import math
import Box2D
from Box2D.Box2D import (
    b2CircleShape, b2Distance, b2PolygonShape, b2Vec2)
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

import logging
logger = logging.getLogger(__name__)

STATE_H = 64
STATE_W = 64
VIEWPORT_W = 512
VIEWPORT_H = 512
FPS = 60
SCALE = 30.0

NUM_OBSTACLES = 3


class BoneDrilling2D(gym.Env, EzPickle):
    metadata = {"render.modes": ["human", "rgb_array"],
                "video.frames_per_second": FPS}

    def __init__(self,
                 pixel_obs=False):
        self.name = 'Simple Navigation'
        self.pixel_obs = pixel_obs
        self.seed()

        # Action space is two floats, [horizontal force, vertical force]
        # Both forces act within the range -1 to +1
        self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)

        if pixel_obs:
            self.observation_space = spaces.Box(
                0, 255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8)
        else:
            # Observation space is distance in X and Y to Goal and Obstacles
            self.observation_space = spaces.Box(
                -np.inf, np.inf, shape=(NUM_OBSTACLES * 2 + 2,),
                dtype=np.float32)

        self.viewer = None

        self.world = Box2D.b2World(gravity=(0, 0), doSleep=True)
        self.obstacles = []
        self.walls = []
        self.agent_start_pos = (0, 0)
        self.goal_pos = (0, 0)
        self.agent = None
        self.done = None
        self.episode_steps = 0
        self.horizon = 500

        # Reset during initialization
        self.reset()

    def step(self, action):
        action = np.clip(action, -1, +1).astype(np.float32).tolist()
        # Apply force to the agent
        # TODO(mustafa): Should we apply noise?
        self.agent.ApplyForceToCenter(force=b2Vec2(action), wake=True)

        # Box2D world simulation step (Next State)
        timeStep = 1.0 / 60
        vel_iters, pos_iters = 6, 2
        self.world.Step(timeStep, vel_iters, pos_iters)

        self.episode_steps += 1
        self.done = self.episode_steps >= self.horizon

        # TODO(mustafa): Currently only returning vector-based obs,
        # rgb_array obs will come from render function, might need to rewrite
        # the step/render loop
        state = []
        distances = []

        for obj in [self.goal] + self.obstacles:
            distance = b2Distance(
                shapeA=self.agent.fixtures[0].shape,
                transformA=self.agent.fixtures[0].body.transform,
                shapeB=obj.fixtures[0].shape,
                transformB=obj.fixtures[0].body.transform
                )
            # Append the X and Y distance seperately
            state.append(abs(distance.pointA[0] - distance.pointB[0]))
            state.append(abs(distance.pointA[1] - distance.pointB[1]))
            distances.append(distance.distance)

        denom = math.hypot(
            self.agent_start_pos[0] - self.goal_pos[0],
            self.agent_start_pos[1] - self.goal_pos[1],
            )
        # Normalized reward based on distance from goal (and original distance)
        # TODO(mustafa): This is a slightly wrong, as moving away will case
        # distance > 1 so not actually normalized. Maybe we use max possible?
        reward = distances[0] / denom

        return np.array(state, dtype=np.float32), reward, self.done, {}

    def reset(self, random_start=False, random_goal=False):

        # Box2D memory management
        self._destroy()

        # Create and position agent
        self.agent_start_pos = self.np_random.rand(2) if random_start else (2, 2)
        self.agent = self.world.CreateDynamicBody(
            shapes=b2CircleShape(pos=self.agent_start_pos, radius=0.5)
            )
        self.agent.color = (1, 0, 0)  # Red

        # Create and position goal
        self.goal_pos = self.np_random.rand(2) if random_goal else (15, 15)
        self.goal = self.world.CreateStaticBody(
            shapes=b2CircleShape(pos=self.goal_pos, radius=0.5)
        )
        self.goal.color = (0, 1, 0)  # Green

        # Generate the obstacles
        for obstacle in range(NUM_OBSTACLES):
            obs_pos = self.np_random.rand(2) * SCALE / 2
            new_obstacle = self.world.CreateStaticBody(
                shapes=b2CircleShape(pos=obs_pos, radius=0.5)
                )
            new_obstacle.color = (1, 1, 0.5)
            self.obstacles.append(new_obstacle)

        wall_size = [(18, 0.1), (0.1, 18), (18, 0.1), (0.1, 18)]
        wall_pos = [(0, 0), (0, 0), (0, 17), (17, 0)]
        for size, pos in zip(wall_size, wall_pos):
            new_wall = self.world.CreateStaticBody(
                shapes=b2PolygonShape(box=size),
                position=pos
            )
            new_wall.color = (0, 0, 0)
            self.walls.append(new_wall)

        # TODO(mustafa): we might not need a drawlist, all bodies are stored
        # in the world object, but just storing them all explicity for now

        # Add all render objects to drawlist to loop over during
        self.drawlist = [self.agent, self.goal] + self.obstacles + self.walls

        print('Environment has been reset!')

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W/SCALE, 0, VIEWPORT_H/SCALE)

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is b2CircleShape:
                    t = rendering.Transform(translation=trans * f.shape.pos)
                    self.viewer.draw_circle(
                        f.shape.radius, 20, color=obj.color
                    ).add_attr(t)
                else:
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color)

        return self.viewer.render(return_rgb_array=(mode == "rgb_aray"))

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed

    def _destroy(self):
        # Destroy all Box2D related entities to manage the memory efficiently
        if not self.agent:
            return

        self.world.DestroyBody(self.agent)
        self.world.DestroyBody(self.goal)
        for ob in self.obstacles:
            self.world.DestroyBody(ob)

        for wall in self.walls:
            self.world.DestroyBody(wall)

        self.obstacles = []
        self.walls = []
        self.agent = None
        self.goal = None


def demo_simple_navigation(env, render=False):
    total_reward = 0

    state = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        state, reward, done, info = env.step((action))
        total_reward += reward

        if render:
            env.render()

        if done:
            break

    env.close()
    print(total_reward)


if __name__ == "__main__":
    demo_simple_navigation(BoneDrilling2D(), render=True)
