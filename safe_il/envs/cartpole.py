import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import torch
import torch.nn as nn 


class CartPole(gym.Env):
    """OpenAI's cartpole env 
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self, seed=None, max_steps=250, normalized_action=False, **kwargs):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"
        self.force_mag = 10.0

        # custom args 
        self.max_steps = max_steps
        self.normalized_action = normalized_action

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        if self.normalized_action:
            self.action_space = spaces.Box(-1, 1, dtype=np.float32)
            self.action_scale = self.force_mag
        else:
            self.action_space = spaces.Box(-self.force_mag, self.force_mag, dtype=np.float32)
            self.action_scale = 1
        self.preprocess_action = lambda a: np.clip(
            a, self.action_space.low, self.action_space.high
        ) * self.action_scale 
            
        self.seed(seed)
        self.viewer = None
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        return [seed]
    
    def reset(self):
        self.step_counter = 0 
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        return np.array(self.state, dtype=np.float32), {}

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        x, x_dot, theta, theta_dot = self.state
        force = float(self.preprocess_action(action))
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot ** 2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.step_counter += 1 
        self.state = np.asarray([x, x_dot, theta, theta_dot])

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        info = {}
        
        # early termination due to time limit 
        max_steps_reached = self.step_counter >= self.max_steps
        if not done and max_steps_reached:
            info["TimeLimit.truncated"] = True 
        done = done or max_steps_reached
            
        if not done:
            reward = 1.0
        else:
            reward = 0.0

        return np.array(self.state, dtype=np.float32), reward, done, info

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = (
                -polewidth / 2,
                polewidth / 2,
                polelen - polewidth / 2,
                -polewidth / 2,
            )
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0.8, 0.6, 0.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
            
    def cost(self, obs, act, info):
        """Per-step cost function for cartpole env.
    
        `obs`, `act` both have shape (*, O or A).
        `info` is a single dict for thet current time step.
        """
        x, theta = obs[:, 0], obs[:, 2]
        length = self.length
        # shape (*, 2)
        ee_pos = torch.stack([x + length * torch.sin(theta), length * torch.cos(theta)], -1)
        goal_pos = torch.as_tensor([0.0, length])
        # shape (*,)
        cost = -torch.exp(-torch.sum(torch.square(ee_pos - goal_pos) * torch.FloatTensor([1.0, 1.0]), -1) / length**2)
        # cost += 0.0001 * torch.sum(torch.square(act), -1)
        return cost
    
    @property
    def model(self):
        """Dynamics function (prior) for agent planning."""
        if not hasattr(self, "_model"):
            self._model = DynamicsModel(self)
        return self._model
     
     
     
class DynamicsModel(nn.Module):
    """Batched cartpole transition dynamics (default euler integration)."""
    
    def __init__(self, env):
        super().__init__()
        self.l = env.length
        self.m = env.masspole
        self.M = env.masscart
        self.g = env.gravity
        self.dt = env.tau
        
        self.Mm = self.m + self.M
        self.ml = self.m * self.l
        self.action_scale = 10 if env.normalized_action else 1

    def forward(self, state, action):
        """Forward cartpole dynamics, two modes (single env or batched).
        
        inputs: (O,), (A,) & output: (O,), or 
        inputs: (*, O), (*, A) & output: (*, O).
        """
        # process state
        if len(state.shape) == 2:
            x, x_dot = state[:, 0], state[:, 1]
            theta, theta_dot = state[:, 2], state[:, 3]
        else:
            x, x_dot = state[0], state[1]
            theta, theta_dot = state[2], state[3]
        # process action
        if len(action.shape) == 2:
            action = action.squeeze(-1)
        else:
            action = action.squeeze()

        temp_factor = (action * self.action_scale
                    + self.ml * theta_dot**2 * torch.sin(theta)) / self.Mm
        theta_dot_dot = (
            (self.g * torch.sin(theta) - torch.cos(theta) * temp_factor) /
            (self.l * (4.0 / 3.0 - self.m * torch.cos(theta)**2 / self.Mm)))
        state_dot = torch.stack([
            x_dot,
            temp_factor - self.ml * theta_dot_dot * torch.cos(theta) / self.Mm,
            theta_dot, theta_dot_dot
        ], -1)

        next_state = state + state_dot * self.dt
        return next_state
