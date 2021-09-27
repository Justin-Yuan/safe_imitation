from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as stats


# -----------------------------------------------------------------------------------
#                   Agent
# -----------------------------------------------------------------------------------

class MPCAgent:
    """Encapsulates sampling-based MPC."""

    def __init__(self,
                 obs_space,
                 act_space,
                 env_cost_func,
                 model,
                 act_opt_freq=1,
                 horizon=25,
                 cem_args={},
                 **kwargs):
        # params
        self.obs_space = obs_space
        self.act_space = act_space
        # NOTE: should match the reward/cost func from `env.step(...)`
        self.env_cost_func = env_cost_func

        # NOTE: determines how often the action sequence will be optimized
        # NOTE: reoptimizes at every call to `act(...)`
        self.act_opt_freq = act_opt_freq
        self.horizon = horizon

        # model
        self.model = model
        self.device = "cpu"

        # planner
        self.dO = obs_space.shape[0]
        self.dU = act_space.shape[0]
        self.ac_ub = act_space.high
        self.ac_lb = act_space.low

        # optimizer (planner)
        self.planner_opt = CEMOptimizer(
            self.horizon * self.dU,
            lower_bound=np.tile(self.ac_lb, [self.horizon]),
            upper_bound=np.tile(self.ac_ub, [self.horizon]),
            **cem_args)

    def to(self, device):
        """Puts agent to device."""
        self.model.to(device)
        self.device = device

    def train(self):
        """Sets training mode."""
        self.model.train()

    def eval(self):
        """Sets evaluation mode."""
        self.model.eval()

    def state_dict(self):
        """Snapshots agent state."""
        return {
            "model": self.model.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """Restores agent state."""
        self.model.load_state_dict(state_dict["model"])

    def reset(self):
        """Resets this controller (at trajecotry start)."""
        self.ac_buf = np.array([]).reshape(0, self.dU)
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.horizon])
        self.init_var = np.tile(
            np.square(self.ac_ub - self.ac_lb) / 16, [self.horizon])
        self.planner_opt.reset()

    def act(self, obs, t, info):
        """Selects action based on learned dynamics and mpc planning.

        Constructs the cost function for the current step, which is 
        different between steps due to different current obs, also 
        passes other necessary arguments `info` for `env_cost_func`.
        """
        cost_func = partial(self.cost_func, obs=obs, info=info)
        action = self._solve_mpc(cost_func)
        return action

    def _solve_mpc(self, cost_func):
        """Solves the MPC optimization problem for action sequence."""
        if self.ac_buf.shape[0] > 0:
            action, self.ac_buf = self.ac_buf[0], self.ac_buf[1:]
            return action

        soln = self.planner_opt.obtain_solution(self.prev_sol, self.init_var,
                                                cost_func)
        # for next call of `act(...)`
        # previous soln is everything after currently taken action
        self.prev_sol = np.concatenate([
            np.copy(soln)[self.act_opt_freq * self.dU:],
            np.zeros(self.act_opt_freq * self.dU)
        ])
        # current soln, can take 1st step as action
        # saves `act_opt_freq` steps to reduce solving mpc every step
        self.ac_buf = soln[:self.act_opt_freq * self.dU].reshape(-1, self.dU)

        return self._solve_mpc(cost_func)

    @torch.no_grad()
    def cost_func(self, ac_seqs, obs=None, info=None):
        """MPC rollout cost.
        
        Args:
            ac_seqs (np.array): decision vars, (pop_size, horizon * act_dim) actions.
            obs (np.array): conditional vars, (O,) current observation.
            info (dict): conditional vars, current info from env.

        Returns:
            np.array: (pop_size,) costs
        """
        pop_size = ac_seqs.shape[0]
        # For parallel compute, (H, Pop_size, A)
        ac_seqs = torch.from_numpy(ac_seqs).float()
        ac_seqs = ac_seqs.view(-1, self.horizon, self.dU).transpose(0, 1)

        # current observation, (Pop_size, O)
        cur_obs = torch.from_numpy(obs).float()
        cur_obs = cur_obs.unsqueeze(0).repeat((pop_size, 1))

        costs = torch.zeros(pop_size)
        for t in range(self.horizon):
            cur_acs = ac_seqs[t]
            # maybe model forward in GPU but mpc planning in CPU
            next_obs = self.model(cur_obs.to(self.device),
                                  cur_acs.to(self.device))
            next_obs = next_obs.cpu()
            cur_obs = next_obs
            # shape (*,)
            cost = self.env_cost_func(next_obs, cur_acs, info)
            costs += cost

        # replace nan with high cost
        costs[costs != costs] = 1e6
        # (Pop_size,)
        return costs.detach().cpu().numpy()


# -----------------------------------------------------------------------------------
#                   Sampling-based Optimizer
# -----------------------------------------------------------------------------------

class CEMOptimizer:
    """Cross-entropy method for (gradient-free) optimization."""

    def __init__(self, sol_dim, lower_bound=None, upper_bound=None, max_iters=5, pop_size=400, num_elites=40, epsilon=0.001, alpha=0.25):
        """Creates the CEM optimizer.

        Args:
            sol_dim (int): The dimensionality of the problem space
            lower_bound (np.array): An array of lower bounds
            upper_bound (np.array): An array of upper bounds
            max_iters (int): The maximum number of iterations to perform during optimization
            pop_size (int): The number of candidate solutions to be sampled at every iteration
            num_elites (int): The number of top solutions that will be used to obtain the distribution
                at the next iteration.
            epsilon (float): A minimum variance. If the maximum variance drops below epsilon, optimization is
                stopped.
            alpha (float): Controls how much of the previous mean and variance is used for the next iteration.
                next_mean = alpha * old_mean + (1 - alpha) * elite_mean, and similarly for variance.
        """
        self.sol_dim = sol_dim
        self.max_iters = max_iters
        self.pop_size = pop_size
        self.num_elites = num_elites

        self.lb = lower_bound
        self.ub = upper_bound
        self.epsilon = epsilon
        self.alpha = alpha

        if num_elites > pop_size:
            raise ValueError("Number of elites must be at most the population size.")

    def reset(self):
        pass

    def obtain_solution(self, init_mean, init_var, cost_func):
        """Optimizes the cost function using the provided initial candidate distribution

        Args:
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            init_var (np.ndarray): The variance of the initial candidate distribution.
            cost_func (callable): cost function with only the unsolved variable as arguments.
        """
        mean, var, t = init_mean, init_var, 0
        X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(var))

        while (t < self.max_iters) and np.max(var) > self.epsilon:
            lb_dist, ub_dist = mean - self.lb, self.ub - mean
            constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)

            samples = X.rvs(size=[self.pop_size, self.sol_dim]) * np.sqrt(constrained_var) + mean
            samples = samples.astype(np.float32)
            costs = cost_func(samples)

            elites = samples[np.argsort(costs)][:self.num_elites]

            new_mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)

            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            var = self.alpha * var + (1 - self.alpha) * new_var

            t += 1
        return mean
    
    
# -----------------------------------------------------------------------------------
#                   Dynamics
# -----------------------------------------------------------------------------------

# TODO: should get this from the env 
class DynamicsModel(nn.Module):
    """Batched cartpole transition dynamics."""

    def __init__(self, env):
        super().__init__()

        self.obs_dim = 4
        self.act_dim = 1

        l, m, M = env.PRIOR_EFFECTIVE_POLE_LENGTH, env.PRIOR_POLE_MASS, env.PRIOR_CART_MASS
        self.l = l
        self.m = m
        self.M = M
        self.Mm = m + M
        self.ml = m * l
        self.g = env.GRAVITY_ACC
        self.dt = env.CTRL_TIMESTEP

        self.force_mag = 10 if env.NORMALIZED_RL_ACTION_SPACE else 1

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

        temp_factor = (action * self.force_mag
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
