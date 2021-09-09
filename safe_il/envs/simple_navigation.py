"""
Simple navigation environment with obstacles to avoid to reach the goal.

To demo the environment and visualize the randomized scene try:

python safe_il/envs/simple_navigation.py
"""

import sys
import pymunk
import pymunk.pygame_util
import pygame
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

import logging
logger = logging.getLogger(__name__)

VIEWPORT_W = 600
VIEWPORT_H = 600
FPS = 60
NUM_OBSTACLES = 3


class SimpleNavigation(gym.Env, EzPickle):
    metadata = {"render.modes": ["human", "rgb_array"],
                "video.frames_per_second": FPS}

    def __init__(self, pixel_obs=False):
        self.name = 'Simple Navigation'
        self.pixel_obs = pixel_obs
        self.seed()

        # Action space is two floats, [horizontal force, vertical force]
        # Both forces act within the range -1 to +1
        self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)

        if pixel_obs:
            self.observation_space = spaces.Box(
                0, 255, shape=(VIEWPORT_W, VIEWPORT_H, 3), dtype=np.uint8)
        else:
            # Observation space is distance in X and Y to Goal and Obstacles
            # [X, Y, for each Obstacle and goal]
            num_obs = (NUM_OBSTACLES * 2) + 2
            self.observation_space = spaces.Box(
                -np.inf, np.inf, shape=(num_obs,),
                dtype=np.float32)

        # PyGame variables
        self.screen = None
        self.clock = None
        self.font = None
        self.options = None

        # Create PyMunk world
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)

        self.obstacles = []
        self.walls = []
        self.goal = None
        self.agent = None
        self.done = False
        self.episode_steps = 0
        self.horizon = 500
        self.shapes = []

    def step(self, action):
        action = np.clip(action, -1, +1).astype(np.float32)
        action = (action[0] * VIEWPORT_H, action[1] * VIEWPORT_W)
        # Apply force to the agent
        # TODO(mustafa): Should we apply noise?
        self.agent.apply_force_at_local_point(action)

        # World simulation step (Next State)
        time_step = 1.0 / FPS
        self.space.step(time_step)

        self.episode_steps += 1

        # TODO(mustafa): Currently only returning vector-based obs,
        # rgb_array obs will come from render function, will need to rewrite
        # the step/render loop to grab the current screen image
        state = []
        for obj in [self.agent] + self.obstacles:
            state.append(np.array(obj.position.normalized()))

        reward = self.step_reward()
        cost = self.step_cost()

        return np.array(state, dtype=np.float32), reward, self.done, {
            'cost': cost
        }

    def reset(self, random_start=False, random_goal=False):

        # Memory Management
        self._destroy()

        # Create and position agent using Dynamic Body
        self.agent_start_pos = self.np_random.rand(2) if random_start \
            else (0.1, 0.1)
        mass = 1
        radius = 0.02 * VIEWPORT_H
        moment = pymunk.moment_for_circle(mass, 0, radius)
        self.agent = pymunk.Body(mass, moment)
        self.agent.position = (self.agent_start_pos[0] * VIEWPORT_H,
                               self.agent_start_pos[1] * VIEWPORT_H)
        self.agent.color = (1, 0, 0)  # Red
        shape = pymunk.Circle(self.agent, radius)
        self.space.add(self.agent, shape)
        self.shapes.append(shape)

        # Create and position goal using Static Body
        self.goal_pos = self.np_random.rand(2) if random_start \
            else (0.9, 0.9)
        radius = 0.02 * VIEWPORT_H
        self.goal = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.goal.position = (self.goal_pos[0] * VIEWPORT_H,
                              self.goal_pos[1] * VIEWPORT_H)
        self.goal.color = (0, 1, 0)  # Green
        shape = pymunk.Circle(self.goal, radius)
        self.space.add(self.goal, shape)
        self.shapes.append(shape)

        # Generate the obstacles using Static Bodies
        for i in range(NUM_OBSTACLES):
            obs_pos = self.np_random.rand(2)
            obs_pos = (obs_pos[0] * VIEWPORT_H, obs_pos[1] * VIEWPORT_H)
            radius = 0.02 * VIEWPORT_H
            new_obstacle = pymunk.Body(body_type=pymunk.Body.STATIC)
            new_obstacle.position = obs_pos
            new_obstacle.color = (1, 1, 0.5)
            shape = pymunk.Circle(new_obstacle, radius)
            self.space.add(new_obstacle, shape)
            self.obstacles.append(new_obstacle)
            self.shapes.append(shape)

        wall_points = [[(1, 1), (1, 599)],
                       [(1, 1), (599, 1)],
                       [(1, 599), (599, 599)],
                       [(599, 1), (599, 599)]]

        for i in range(4):
            line_shape = pymunk.Segment(
                self.space.static_body,
                wall_points[i][0],

                wall_points[i][1],
                0.0)
            line_shape.friction = 0.99
            self.walls.append(line_shape)
            self.space.add(line_shape)

        # TODO(mustafa): we might not need a drawlist, all bodies are stored
        # in the world object, but just storing them all explicity for now

        # Add all render objects to drawlist to loop over during
        self.drawlist = [self.agent, self.goal] + self.obstacles + self.walls

        print('Environment has been reset!')

    def step_reward(self):
        """
        Reward function based on distance from goal
        """
        reward = np.linalg.norm(
            np.subtract(self.agent.position.normalized(),
                        self.goal.position.normalized()))
        print(reward)
        return reward

    def step_cost(self):
        """
        Cost function that takes into account the distance to the obstacles
        """
        cost = 0
        return cost

    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 16)
            pymunk.pygame_util.positive_y_is_up = True
            self.options = pymunk.pygame_util.DrawOptions(self.screen)

        self.screen.fill(pygame.Color("white"))
        self.space.debug_draw(self.options)
        pygame.display.flip()
        self.clock.tick()
        pygame.display.set_caption("fps: " + str(self.clock.get_fps()))

        return 0

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed

    def _destroy(self):
        # Destroy all related entities to manage the memory efficiently
        if not self.agent:
            return

        self.space.remove(self.agent)
        self.agent = None
        self.space.remove(self.goal)
        self.goal = None

        for s in self.shapes:
            self.space.remove(s)

        for w in self.walls:
            self.space.remove(w)

        self.shapes = []
        self.walls = []


def demo_simple_navigation(env, render=False, manual_control=False):
    total_reward = 0

    state = env.reset()
    for _ in range(10000):
        action = [0.0, 0.0]
        if manual_control:
            if env.screen is None:
                pygame.init()
                env.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
                env.clock = pygame.time.Clock()
                env.font = pygame.font.SysFont("Arial", 16)
                pymunk.pygame_util.positive_y_is_up = True
                env.options = pymunk.pygame_util.DrawOptions(env.screen)
            # Capture keyboard events
            for event in pygame.event.get():
                if (event.type == pygame.KEYDOWN and
                        event.key == pygame.K_RIGHT):
                    action[0] = 1.0

                if (event.type == pygame.KEYDOWN and
                        event.key == pygame.K_LEFT):
                    action[0] = -1.0

                if (event.type == pygame.KEYDOWN and
                        event.key == pygame.K_UP):
                    action[1] = 1.0

                if (event.type == pygame.KEYDOWN and
                        event.key == pygame.K_DOWN):
                    action[1] = -1.0
        else:
            # Random action
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
    env = gym.make("safe_il:SimpleNavigation-v0")
    sys.exit(demo_simple_navigation(env, render=True, manual_control=True))
