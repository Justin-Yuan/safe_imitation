"""
Bone Drilling environment to reach the goal and avoid obstacles, with safety
constraints in regards to temperature that can damage nearby structures.

To demo the environment and visualize the randomized scene try:

python safe_il/envs/bone_drilling_2d.py
"""
import sys
import pymunk
import pymunk.autogeometry
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
RADIUS = 0.02


def generate_geometry(surface, space):
    for s in space.shapes:
        if hasattr(s, "generated") and s.generated:
            space.remove(s)

    def sample_func(point):
        try:
            p = int(point[0]), int(point[1])
            color = surface.get_at(p)
            return 0 if color.b == 255 else 1  # use lightness
        except Exception as e:
            print(e)
            return 0

    line_set = pymunk.autogeometry.march_soft(
        pymunk.BB(0, 0, 599, 599), 60, 60, 0.5, sample_func
    )

    for polyline in line_set:
        line = pymunk.autogeometry.simplify_curves(polyline, 1.0)

        for i in range(len(line) - 1):
            p1 = line[i]
            p2 = line[i + 1]
            shape = pymunk.Segment(space.static_body, p1, p2, 1)
            shape.friction = 0.5
            shape.color = pygame.Color("red")
            shape.generated = True
            space.add(shape)


class BoneDrilling2D(gym.Env, EzPickle):
    metadata = {"render.modes": ["human", "rgb_array"],
                "video.frames_per_second": FPS}

    def __init__(self, pixel_obs=False):
        self.name = 'Bone Drilling'
        self.is_pixel_obs = pixel_obs
        self.seed()

        # Action space is two floats, [horizontal force, vertical force]
        # Both forces act within the range -1 to +1
        self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)

        if pixel_obs:
            self.observation_space = spaces.Box(
                0, 255, shape=(VIEWPORT_W, VIEWPORT_H, 3), dtype=np.uint8)
        else:
            # For vector-based observation of the state, we measure the
            # distance to each obstacle and the goal. We also measure the
            # temperature of the obstacles.
            # [X & Y distance for each obstacle, X & Y for goal, Temp for each]
            num_obs = (NUM_OBSTACLES * 2) + 2 + NUM_OBSTACLES
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
        self.bone = None

        self.render()

    def step(self, action):
        action = np.clip(action, -1, +1).astype(np.float32)
        action = (action[0] * VIEWPORT_H, action[1] * VIEWPORT_W)
        # Apply force to the agent
        # TODO(mustafa): Should we apply noise?
        self.agent.apply_force_at_local_point(action)

        pygame.draw.circle(
            self.bone, pygame.Color("white"),
            self.agent.position,
            0.05 * VIEWPORT_H)  # Slightly larger than the agent
        generate_geometry(self.bone, self.space)

        # World simulation step (Next State)
        time_step = 1.0 / FPS
        self.space.step(time_step)

        self.episode_steps += 1

        state = []
        for obj in [self.agent] + [self.goal] + self.obstacles:
            x, y = obj.position
            position = [x / VIEWPORT_W, y / VIEWPORT_W]
            state.append(position)

        reward = self.step_reward(state)
        cost = self.step_cost(state)

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
        radius = RADIUS * VIEWPORT_H
        moment = pymunk.moment_for_circle(mass, 0, radius)
        self.agent = pymunk.Body(mass, moment)
        self.agent.position = (self.agent_start_pos[0] * VIEWPORT_H,
                               self.agent_start_pos[1] * VIEWPORT_H)
        shape = pymunk.Circle(self.agent, radius)
        shape.color = pygame.Color("black")
        self.space.add(self.agent, shape)
        self.shapes.append(shape)

        # Create and position goal using Static Body
        self.goal_pos = self.np_random.rand(2) if random_start \
            else (0.9, 0.9)
        radius = RADIUS * VIEWPORT_H
        self.goal = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.goal.position = (self.goal_pos[0] * VIEWPORT_H,
                              self.goal_pos[1] * VIEWPORT_H)
        shape = pymunk.Circle(self.goal, radius)
        shape.color = pygame.Color("green")
        self.space.add(self.goal, shape)
        self.shapes.append(shape)

        # Generate the obstacles using Static Bodies
        for i in range(NUM_OBSTACLES):
            obs_pos = self.np_random.rand(2)
            obs_pos = (obs_pos[0] * VIEWPORT_H, obs_pos[1] * VIEWPORT_H)
            radius = RADIUS * VIEWPORT_H
            new_obstacle = pymunk.Body(body_type=pymunk.Body.STATIC)
            new_obstacle.position = obs_pos
            shape = pymunk.Circle(new_obstacle, radius)
            shape.color = pygame.Color("red")
            self.space.add(new_obstacle, shape)
            self.obstacles.append(new_obstacle)
            self.shapes.append(shape)

        wall_points = [[(1, 1), (1, VIEWPORT_H)],
                       [(1, 1), (VIEWPORT_W, 1)],
                       [(1, VIEWPORT_H), (VIEWPORT_W, VIEWPORT_H)],
                       [(VIEWPORT_W, 1), (VIEWPORT_W, VIEWPORT_H)]]

        for i in range(4):
            line_shape = pymunk.Segment(
                self.space.static_body,
                wall_points[i][0],
                wall_points[i][1],
                0.0)
            line_shape.friction = 0.99
            self.walls.append(line_shape)
            self.space.add(line_shape)

        print('Environment has been reset!')

    def step_reward(self, state):
        """
        Reward function based on distance from agent to goal normalized between
        [0,1] using the maximum possible distance, and taking into account 
        radius of both bodies
        """
        distance = np.linalg.norm(np.subtract(state[0], state[1]))
        distance = distance - (RADIUS * 2)
        reward = 1 - (distance / np.linalg.norm(np.subtract([0, 0], [1, 1])))
        reward = np.clip(reward, 0.0, 1.0)
        print(reward)
        return reward

    def step_cost(self, state):
        """
        Cost function that takes into account the distance to the obstacles
        and their temperatures
        """
        cost = 0
        return cost

    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 16)
            pymunk.pygame_util.positive_y_is_up = False
            self.options = pymunk.pygame_util.DrawOptions(self.screen)

        if self.bone is None:
            self.bone = pygame.Surface((600, 600))
            self.bone.fill(pygame.Color("yellow"))

        self.screen.fill(pygame.Color("white"))
        self.screen.blit(self.bone, (0, 0))

        self.space.debug_draw(self.options)
        pygame.display.flip()
        self.clock.tick(FPS)
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


def demo_bone_drilling_2d(env, render=False, manual_control=False):
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

                # Y is flipped in Bone Drilling rendering
                if (event.type == pygame.KEYDOWN and
                        event.key == pygame.K_UP):
                    action[1] = -1.0

                if (event.type == pygame.KEYDOWN and
                        event.key == pygame.K_DOWN):
                    action[1] = 1.0
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
    env = gym.make("safe_il:BoneDrilling2D-v0")
    sys.exit(demo_bone_drilling_2d(env, render=True, manual_control=True))
