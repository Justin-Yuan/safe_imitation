"""
Bone Drilling environment to reach the goal and avoid obstacles, with safety
constraints in regards to temperature that can damage nearby structures.

To demo the environment and visualize the randomized scene try:

python safe_il/envs/bone_drilling_2d.py
"""

import math
import Box2D
from Box2D.Box2D import (
    b2CircleShape, b2ContactListener, b2Distance, b2PolygonShape, b2Vec2)
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

import logging
logger = logging.getLogger(__name__)

STATE_H = 64
STATE_W = 64
VIEWPORT_W = 600
VIEWPORT_H = 600
FPS = 60
SCALE = 30.0
# NOTE(mustafa): The enviornment works on a scaled resolution, 
# so all positing should be scaled to this. Just normalize everything [0,1]
# and then multiply by this constant for now, need to look at this closer after
SCALED_MULTIP = VIEWPORT_H / SCALE

NUM_OBSTACLES = 3


class ContactDetector(b2ContactListener):
    def __init__(self, env):
        b2ContactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        # Check for contact with goal for task completion
        # NOTE(mustafa): either FixtureA or FixtureB of the contact could
        # be the the goal/agent, so we have to check everything both ways
        bodies = [contact.fixtureA.body, contact.fixtureB.body]
        if self.env.agent in bodies and self.env.goal in bodies:
            self.env.game_over = True

        # Check contact on bone structures

    def EndContact(self, contact):
        pass

    def PreSolve(self, contact, oldManifold):
        pass

    def PostSolve(self, contact, impulse):
        pass


class BoneDrilling2D(gym.Env, EzPickle):
    metadata = {"render.modes": ["human", "rgb_array"],
                "video.frames_per_second": FPS}

    def __init__(self,
                 pixel_obs=False):
        self.name = 'Bone Drilling'
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

        self.world = Box2D.b2World(gravity=(0, 0))

        self.obstacles = []
        self.walls = []
        self.agent_start_pos = (0, 0)
        self.goal_pos = (0, 0)
        self.agent = None
        self.done = None
        self.episode_steps = 0
        self.horizon = 10000

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

        # Necessary to clear forces in Box2D after sim step
        # TODO(mustafa): Important! The forces might be cleared but velocity
        # is maintained, need to fix this if its not intended
        self.world.ClearForces()

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

        # Reset the Box2D Contact Listener
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False

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
            obs_pos = self.np_random.rand(2) * SCALED_MULTIP
            new_obstacle = self.world.CreateStaticBody(
                shapes=b2CircleShape(pos=obs_pos, radius=0.5)
                )
            new_obstacle.color = (1, 1, 0.5)
            self.obstacles.append(new_obstacle)

        wall_size = [(SCALED_MULTIP, 0.1), (0.1, SCALED_MULTIP),
                     (SCALED_MULTIP, 0.1), (0.1, SCALED_MULTIP)]
        wall_pos = [(0, 0), (0, 0), (0, SCALED_MULTIP), (SCALED_MULTIP, 0)]
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

        self.world.contactListener = None
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


def demo_bone_drilling_2d(env, render=False, manual_control=False):
    # Key navigation borrowed from car_racing.py in gym
    from pyglet.window import key

    a = np.array([0.0, 0.0])

    def key_press(k, mod):
        global restart
        if k == 0xFF0D:
            restart = True
        if k == key.LEFT:
            a[0] = -1.0
        if k == key.RIGHT:
            a[0] = +1.0
        if k == key.UP:
            a[1] = +1.0
        if k == key.DOWN:
            a[1] = -1.0

    def key_release(k, mod):
        if k == key.LEFT and a[0] == -1.0:
            a[0] = 0
        if k == key.RIGHT and a[0] == +1.0:
            a[0] = 0
        if k == key.UP and a[1] == +1.0:
            a[1] = 0
        if k == key.DOWN and a[1] == -1.0:
            a[1] = 0

    if manual_control:
        env.render()
        env.viewer.window.on_key_press = key_press
        env.viewer.window.on_key_release = key_release

    record_video = False
    if record_video:
        from gym.wrappers.monitor import Monitor
        env = Monitor(env, "/tmp/video-test", force=True)

    total_reward = 0

    state = env.reset()
    for _ in range(1000):
        if manual_control:
            action = a
            print(a)
        else:
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
    env = gym.make("safe_il:BoneDrilling2D-v0")

    demo_bone_drilling_2d(env, render=True, manual_control=True)
