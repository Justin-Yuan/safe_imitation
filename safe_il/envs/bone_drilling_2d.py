"""
Bone Drilling environment to reach the goal and avoid obstacles, with safety
constraints in regards to temperature that can damage nearby structures.

To demo the environment and visualize the randomized scene try:

python safe_il/envs/bone_drilling_2d.py
"""

import time
import math
import Box2D
from Box2D.Box2D import (
    b2CircleShape, b2ContactFilter, b2ContactListener, b2Distance, b2PolygonShape, b2Vec2)
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
# so all positioning should be scaled to this. Just normalize everything [0,1]
# and then multiply by this constant for now, need to look at this closer after
SCALED_MULTIP = VIEWPORT_H / SCALE

NUM_OBSTACLES = 3
TEMP_GRID_SIZE = int(SCALED_MULTIP)
BONE_GRID_SIZE = int(SCALED_MULTIP) - 4
BONE_CELL_LENGTH = 1

'''
NOTE(mustafa):
Currently using a simple grid structure of bones, coming into contact with the
bones rigidbody removes that body from the world. Might change to custom
polygon shapes in the future, allowing for custom geometry that matches the
shape of the drillbit

Temperature is an attribute of the bodies, and is calculated based on distance
as opposed to the previous idea of using a temperature grid and incrementing
cells around the heat source. Heat is now measured as a function based on
distance from the drill's rigid body. This system holds under the assumption
that everything is surrounded by bone, and heat transfer is constant from the
source to object
'''


class BoneData():
    def __init__(self):
        self.flagged_destroy = False


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
        if self.env.agent in bodies:
            if bodies[0] == self.env.agent:
                contact.fixtureB.body.userData.flagged_destroy = True
            else:
                contact.fixtureA.body.userData.flagged_destroy = True

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
        self.bones = []
        self.temperature_grid = np.zeros(
            shape=(TEMP_GRID_SIZE, TEMP_GRID_SIZE), dtype=np.uint8)
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

        # During the step we can't destroy bodies, so we destroy them now
        for bone in self.bones:
            if bone.userData.flagged_destroy:
                self.world.DestroyBody(bone)
                self.bones.remove(bone)

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

            # Increment the temperature
            # TODO(mustafa): currently assumes drill always on, also
            # need to find a correct function here from literature
            obj.temperature += 0.01 / distance.distance
            print(obj.temperature)

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
        self.agent_start_pos = self.np_random.rand(2) if random_start \
            else (2, 2)
        self.agent = self.world.CreateDynamicBody(
            shapes=b2CircleShape(pos=self.agent_start_pos, radius=0.5)
            )
        self.agent.color = (1, 0, 0)  # Red
        self.agent.temperature = 0.0

        # Create and position goal
        self.goal_pos = self.np_random.rand(2) if random_goal else (15, 15)
        self.goal = self.world.CreateStaticBody(
            shapes=b2CircleShape(pos=self.goal_pos, radius=0.5)
        )
        self.goal.color = (0, 1, 0)  # Green
        self.goal.temperature = 0.0  # Not necessary, but just being consistent

        # Generate the obstacles
        for obstacle in range(NUM_OBSTACLES):
            obs_pos = self.np_random.rand(2) * SCALED_MULTIP
            new_obstacle = self.world.CreateStaticBody(
                shapes=b2CircleShape(pos=obs_pos, radius=0.5)
                )
            new_obstacle.color = (0.1, 0.1, 0.1)
            new_obstacle.temperature = 0.0
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
            new_wall.temperature = 0.0  # Also not necessary
            self.walls.append(new_wall)

        # Generate the bone structure as a grid
        # NOTE(mustafa): currently using the scaled width of the env
        for x in range(BONE_GRID_SIZE):
            for y in range(BONE_GRID_SIZE):
                new_bone = self.world.CreateStaticBody(
                    shapes=b2PolygonShape(
                        box=(BONE_CELL_LENGTH, BONE_CELL_LENGTH)),
                    position=((BONE_CELL_LENGTH * x) + 4,
                              (BONE_CELL_LENGTH * y) + 4),
                    userData=BoneData()
                )
                new_bone.color = (0.6, 0.6, 0.5)  # Beige Color
                new_bone.temperature = 0.0
                self.bones.append(new_bone)

        # TODO(mustafa): we might not need a drawlist, all bodies are stored
        # in the world object, but just storing them all explicity for now

        # Add all render objects to drawlist to loop over during
        self.drawlist = [self.agent, self.goal] + self.walls + self.bones + \
            self.obstacles

        print('Environment has been reset!')

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W/SCALE, 0, VIEWPORT_H/SCALE)

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                # Change the red tone of color based on temperature
                color = (
                        np.clip(obj.color[0] + obj.temperature, 0.0, 1.0),
                        obj.color[1],
                        obj.color[2])
                if type(f.shape) is b2CircleShape:
                    t = rendering.Transform(translation=trans * f.shape.pos)
                    self.viewer.draw_circle(
                        f.shape.radius, 20, color=color
                    ).add_attr(t)
                else:
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=color)

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

        for bone in self.bones:
            self.world.DestroyBody(bone)

        self.obstacles = []
        self.walls = []
        self.bones = []
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
    for _ in range(10000):
        if manual_control:
            action = a
        else:
            action = env.action_space.sample()
        
        step_t_start = time.time()
        state, reward, done, info = env.step(action)
        step_t_end = time.time()
        total_reward += reward

        if render:
            render_t_start = time.time()
            env.render()
            render_t_end = time.time()

        if done:
            break
        
        if _ % 20 == 0:
            print(f"Step time: {step_t_end - step_t_start}s")
            print(f"Render time: {render_t_end - render_t_start}s")

    env.close()
    print(total_reward)


if __name__ == "__main__":
    env = gym.make("safe_il:BoneDrilling2D-v0")

    demo_bone_drilling_2d(env, render=True, manual_control=True)
