from collections import defaultdict
from functools import partial
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
import scipy.stats as stats

from safe_il.utils import random_sample

# -----------------------------------------------------------------------------------
#                   Agent
# -----------------------------------------------------------------------------------

class PETSAgent:
    """Encapsulates ensemble model and sampling-based MPC."""

    def __init__(self,
                 obs_space,
                 act_space,
                 env_cost_func,
                 hidden_dim=500,
                 ensemble_size=5,
                 weight_decays=[],
                 lr=0.001,
                 epochs=5,
                 batch_size=256,
                 act_opt_freq=1,
                 horizon=25,
                 num_particles=20,
                 cem_args={},
                 **kwargs):
        # params
        self.obs_space = obs_space
        self.act_space = act_space
        # NOTE: should match the reward/cost func from `env.step(...)`
        self.env_cost_func = env_cost_func

        self.ensemble_size = ensemble_size
        self.epochs = epochs
        self.batch_size = batch_size

        # NOTE: determines how often the action sequence will be optimized
        # NOTE: reoptimizes at every call to `act(...)`
        self.act_opt_freq = act_opt_freq
        self.horizon = horizon
        self.num_particles = num_particles
        assert num_particles % ensemble_size == 0, "Number of particles must be a multiple of the ensemble size."
        self.particles_per_ensem = num_particles // ensemble_size

        # model
        self.model = EnsembleModel(ensemble_size,
                                   in_features=obs_space.shape[0] + act_space.shape[0],
                                   out_features=obs_space.shape[0] * 2,
                                   hidden_size=hidden_dim,
                                   num_layers=len(weight_decays),
                                   weight_decays=weight_decays)
        self.device = "cpu"

        # optimizer (model)
        self.model_opt = torch.optim.Adam(self.model.parameters(), lr)

        # planner
        self.dO = obs_space.shape[0]
        self.dU = act_space.shape[0]
        self.ac_ub = act_space.high
        self.ac_lb = act_space.low

        # optimizer (planner)
        self.planner_opt = CEMOptimizer(self.horizon * self.dU,
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
        return {"model": self.model.state_dict(), "model_opt": self.model_opt.state_dict()}

    def load_state_dict(self, state_dict):
        """Restores agent state."""
        self.model.load_state_dict(state_dict["model"])
        self.model_opt.load_state_dict(state_dict["model_opt"])

    def update(self, rollouts, device="cpu"):
        """Performs a training step on ensemble model."""
        resutls = defaultdict(list)
        num_batch = rollouts.num_steps // self.batch_size
        # initial buffer size can be smaller than batch_size
        num_batch = max(num_batch, 1)

        # get normalization heuristics
        train_inputs, _ = rollouts.get(to_torch=False)
        self.model.fit_input_stats(train_inputs)

        # inner training loop
        for epoch in range(self.epochs):
            reg_loss_epoch, nll_loss_epoch, mse_loss_epoch = 0, 0, 0
            sampler = rollouts.sampler(self.batch_size, num_nets=self.ensemble_size, device=device, drop_last=False)
            for train_in, train_targ in sampler:
                # each has shape (N, B, *)
                # regularization loss
                loss = 0.01 * (self.model.max_logvar.sum() - self.model.min_logvar.sum())
                reg_loss = self.model.compute_decays()
                loss += reg_loss
                reg_loss_epoch += reg_loss.item()

                # dynamics (nll) loss
                mean, logvar = self.model(train_in.float(), ret_logvar=True)
                inv_var = torch.exp(-logvar)
                state_loss = ((mean - train_targ)**2) * inv_var + logvar
                state_loss = state_loss.mean(-1).mean(-1).sum()
                loss += state_loss
                nll_loss_epoch += state_loss.item()

                # mse loss (on predicted mean)
                # not used for learning, only to monitor model accuracy
                mse_loss = (mean - train_targ)**2
                mse_loss = mse_loss.detach().mean(-1).mean(-1).sum()
                mse_loss_epoch += mse_loss.item()

                # perform update
                self.model_opt.zero_grad()
                loss.backward()
                self.model_opt.step()

            # `num_batch` is off by a little with sampler `drop_last=False`
            resutls["reg_loss"].append(reg_loss_epoch / num_batch)
            resutls["nll_loss"].append(nll_loss_epoch / num_batch)
            resutls["mse_loss"].append(mse_loss_epoch / num_batch)

        resutls = {k: sum(v) / len(v) for k, v in resutls.items()}
        return resutls

    def reset(self):
        """Resets this controller (at trajecotry start)."""
        self.ac_buf = np.array([]).reshape(0, self.dU)
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.horizon])
        self.init_var = np.tile(np.square(self.ac_ub - self.ac_lb) / 16, [self.horizon])
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

        soln = self.planner_opt.obtain_solution(self.prev_sol, self.init_var, cost_func)
        # for next call of `act(...)`
        # previous soln is everything after currently taken action
        self.prev_sol = np.concatenate([np.copy(soln)[self.act_opt_freq * self.dU:], np.zeros(self.act_opt_freq * self.dU)])
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
        # For parallel compute, (Pop_size, H*A) -> (H, Pop_size * Num_par, A)
        ac_seqs = torch.from_numpy(ac_seqs).float()
        # (H, Pop_size, A)
        ac_seqs = ac_seqs.view(-1, self.horizon, self.dU).transpose(0, 1)
        # (H, Pop_size, Num_par, A)
        ac_seqs = ac_seqs.unsqueeze(2).expand(-1, -1, self.num_particles, -1)
        # (H, Pop_size * Num_par, A)
        ac_seqs = ac_seqs.contiguous().view(self.horizon, -1, self.dU)

        # current observation, (O,) -> (Pop_size * Num_par, O)
        cur_obs = torch.from_numpy(obs).float()
        cur_obs = cur_obs.unsqueeze(0).repeat((pop_size * self.num_particles, 1))

        costs = torch.zeros(pop_size, self.num_particles)
        for t in range(self.horizon):
            cur_acs = ac_seqs[t]
            # maybe model forward in GPU but mpc planning in CPU
            # (Pop_size * Num_par, O) + (Pop_size * Num_par, A) -> (Pop_size * Num_par, O)
            next_obs = self.predict_next_obs(cur_obs, cur_acs)
            next_obs = next_obs.cpu()
            cur_obs = next_obs
            # shape (*,)
            cost = self.env_cost_func(next_obs, cur_acs, info)
            # (Pop_size * Num_par,) -> (Pop_size, Num_par)
            cost = cost.view(-1, self.num_particles)
            costs += cost

        # replace nan with high cost
        costs[costs != costs] = 1e6
        mean_cost = costs.mean(dim=1)
        # (Pop_size,)
        return mean_cost.detach().cpu().numpy()

    def predict_next_obs(self, obs, acs):
        """Get next state from current dynamics model. 
        
        Args:
            obs (torch.FloatTensor): (*, O)
            acs (torch.FloatTensor): (*, A)
            
        Returns:
            torch.FloatTensor: (*, O) next state 
        """
        proc_obs = self._reshape_model_input(obs)
        acs = self._reshape_model_input(acs)

        inputs = torch.cat((proc_obs, acs), dim=-1).to(self.device)
        with torch.no_grad():
            mean, var = self.model(inputs)
        # sample next obs
        predictions = mean + torch.randn_like(mean).to(self.device) * var.sqrt()

        # TS Optimization: Remove additional dimension
        predictions = self._reshape_model_output(predictions)
        next_obs = obs.to(self.device) + predictions
        return next_obs

    def _reshape_model_input(self, x):
        """Converts (Pop_size*Num_par, O) -> (N, *, O)."""
        dim = x.shape[-1]
        new_x = x.reshape(-1, self.ensemble_size, self.particles_per_ensem, dim)
        new_x = new_x.transpose(0, 1).reshape(self.ensemble_size, -1, dim)
        return new_x

    def _reshape_model_output(self, x):
        """Converts (N, *, O) -> (Pop_size*Num_par, O)."""
        dim = x.shape[-1]
        new_x = x.reshape(self.ensemble_size, -1, self.particles_per_ensem, dim)
        new_x = x.transpose(0, 1).reshape(-1, dim)
        return new_x


# -----------------------------------------------------------------------------------
#                   Model
# -----------------------------------------------------------------------------------

class EnsembleModel(nn.Module):
    """Model for a PETS agent."""

    def __init__(self, ensemble_size, in_features, out_features, hidden_size, num_layers, weight_decays):
        super().__init__()

        self.ensemble_size = ensemble_size
        self.in_features = in_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.weight_decays = weight_decays

        self.linear_layers = nn.ParameterList()
        # input layer
        self.linear_layers.extend(get_affine_params(ensemble_size, in_features, hidden_size))
        # hidden layers
        for i in range(num_layers - 2):
            self.linear_layers.extend(get_affine_params(ensemble_size, hidden_size, hidden_size))
        # output layer
        self.linear_layers.extend(get_affine_params(ensemble_size, hidden_size, out_features))

        # input normalization
        self.inputs_mu = nn.Parameter(torch.zeros(1, in_features), requires_grad=False)
        self.inputs_sigma = nn.Parameter(torch.zeros(1, in_features), requires_grad=False)

        # output variance bound
        self.max_logvar = nn.Parameter(torch.ones(1, out_features // 2, dtype=torch.float32) / 2.0)
        self.min_logvar = nn.Parameter(-torch.ones(1, out_features // 2, dtype=torch.float32) * 10.0)

    def compute_decays(self):
        """Gets L2 regularization loss, only for weights (W) not bias (b)."""
        decay = 0
        for layer, weight_decay in zip(self.linear_layers[::2], self.weight_decays):
            decay += weight_decay * (layer**2).sum() / 2.0
        return decay

    def fit_input_stats(self, data):
        """Gets 1st, 2nd moments from data."""
        mu = np.mean(data, axis=0, keepdims=True)
        sigma = np.std(data, axis=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0

        self.inputs_mu.data = torch.from_numpy(mu).to(self.inputs_mu.device).float()
        self.inputs_sigma.data = torch.from_numpy(sigma).to(self.inputs_sigma.device).float()

    def forward(self, inputs, ret_logvar=False):
        """Gets ensemble predictions

        Args:
            inputs (torch.FloatTensor): shape (N,B,I). 
            ret_logvar (bool): if to return log-variance or variance.
            
        Returns:
            3 torch.FloatTensor: (N,B,O) predicted mean, (log-)variance and catastrophe signal.
        """
        # Transform inputs
        # NUM_NETS x BATCH_SIZE X INPUT_LENGTH
        # (N,B,I)
        inputs = (inputs - self.inputs_mu) / self.inputs_sigma

        # (N,B,I) x (N,I,O) -> (N,B,O), (N,B,O1) x (N,O1,O2) -> (N,B,O2)
        for i, layer in enumerate(zip(self.linear_layers[::2], self.linear_layers[1::2])):
            weight, bias = layer
            inputs = inputs.matmul(weight) + bias
            if i < self.num_layers - 1:
                inputs = swish(inputs)

        mean = inputs[:, :, :self.out_features // 2]
        logvar = inputs[:, :, self.out_features // 2:]

        # bound variance output
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_logvar:
            return mean, logvar
        return mean, torch.exp(logvar)


# -----------------------------------------------------------------------------------
#                   Storage
# -----------------------------------------------------------------------------------

class PETSBuffer(object):
    """Storage for rollouts during training (for dynamics model).

    Attributes:
        train_inputs (list): rollouts of training inputs, [(T,O+A)]. 
        train_targets (list): rollouts of training targets, [(T,O)].
        num_rollouts (int): total number of rollouts. 
        num_steps (int): total number of steps.
    """

    def __init__(self, obs_space, act_space, batch_size=None):
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        """Allocate space for containers."""
        self.train_inputs = []
        self.train_targets = []
        self.num_rollouts = 0
        self.num_steps = 0

    def __len__(self):
        """Returns current size of the buffer."""
        return self.num_steps

    def state_dict(self):
        """Packages buffer data."""
        return {key: getattr(self, key) for key in ["train_inputs", "train_targets", "num_rollouts", "num_steps"]}

    def load_state_dict(self, data):
        """Restores past buffer data."""
        for key, val in data.items():
            setattr(self, key, val)

    def push(self, input_batch, target_batch):
        """Inserts transition step data (as dict) to storage.
        
        Args:
            input_batch (list): rollouts of inputs, [(T,O+A)]
            target_batch (list): rollouts of targets, [(T,O)]
        """
        self.train_inputs.extend(input_batch)
        self.train_targets.extend(target_batch)
        self.num_rollouts += len(input_batch)
        self.num_steps += sum([int(traj.shape[0]) for traj in input_batch])

    def get(self, to_torch=False, device="cpu"):
        """Returns all current data."""
        train_inputs = np.concatenate(self.train_inputs, 0)
        train_targets = np.concatenate(self.train_targets, 0)
        # convert to torch tensors if needed
        if to_torch:
            train_inputs = torch.as_tensor(train_inputs, device=device)
            train_targets = torch.as_tensor(train_targets, device=device)
        return train_inputs, train_targets

    def sampler(self, batch_size, num_nets=1, device="cpu", drop_last=False):
        """Makes sampler to loop through all data for ensemble model.
        
        Assumes batch_size B, num_nets N, feature size *,
        Each output is (N, B, *) for ensemble models.
        """
        total_steps = len(self)
        samplers = [random_sample(np.arange(total_steps), batch_size, drop_last) for _ in range(num_nets)]
        train_inputs = np.concatenate(self.train_inputs, 0)
        train_targets = np.concatenate(self.train_targets, 0)

        for indices_list in zip(*samplers):
            inputs = torch.as_tensor([train_inputs[indices] for indices in indices_list], device=device)
            targets = torch.as_tensor([train_targets[indices] for indices in indices_list], device=device)
            yield inputs, targets


# -----------------------------------------------------------------------------------
#                   Misc
# -----------------------------------------------------------------------------------

def swish(x):
    return x * torch.sigmoid(x)


def truncated_normal(size, mean=0, std=1):
    """Truncated normal for pytorch. 
    
    Reference https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/19
    """
    tensor = torch.zeros(size)
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def get_affine_params(ensemble_size, in_features, out_features):
    """Gets weight and bias parameters for ensemble linear layer."""
    w = truncated_normal(size=(ensemble_size, in_features, out_features), std=1.0 / (2.0 * np.sqrt(in_features)))
    w = nn.Parameter(w)
    b = nn.Parameter(torch.zeros(ensemble_size, 1, out_features, dtype=torch.float32))
    return w, b


def truncated_normal2(size, std):
    """reference: https://github.com/abalakrishna123/recovery-rl/blob/master/config/utils.py
    """
    val = stats.truncnorm.rvs(-2, 2, size=size) * std
    return torch.tensor(val, dtype=torch.float32)


def get_affine_params2(ensemble_size, in_features, out_features):
    """reference: https://github.com/abalakrishna123/recovery-rl/blob/master/config/utils.py
    """
    w = truncated_normal2(size=(ensemble_size, in_features, out_features), std=1.0 / (2.0 * np.sqrt(in_features)))
    w = nn.Parameter(w)
    b = nn.Parameter(torch.zeros(ensemble_size, 1, out_features, dtype=torch.float32))
    return w, b


# -----------------------------------------------------------------------------------
#                   Env-specific Hacks
# -----------------------------------------------------------------------------------


def cartpole_cost_func(obs, act, info):
    """Per-step cost function for cartpole env.
    
    `obs`, `act` both have shape (*, O or A).
    `info` is a single dict for thet current time step.
    """
    x, theta = obs[:, 0], obs[:, 2]
    length = info["pendulum_length"]
    # shape (*, 2)
    ee_pos = torch.stack([x + length * torch.sin(theta), length * torch.cos(theta)], -1)
    goal_pos = torch.as_tensor([0.0, length])
    # shape (*,)
    cost = -torch.exp(-torch.sum(torch.square(ee_pos - goal_pos) * torch.FloatTensor([1.0, 1.0]), -1) / length**2)
    # cost += 0.01 * torch.sum(torch.square(act), -1)
    # cost += 0.0001 * torch.sum(torch.square(act), -1)
    return cost


def cartpole_cost_info_func(info, env):
    """Per-step function to augment info dict for cost func.
    """
    info["pendulum_length"] = env.OVERRIDDEN_EFFECTIVE_POLE_LENGTH
    return info


def quadrotor_cost_func(obs, act, info):
    """Per-step cost function for quadrotor env."""
    if obs.shape[-1] == 2:
        z = obs[:, 0]
        z_goal = torch.as_tensor(info["goal"][1])
        cost = (z - z_goal)**2
    elif obs.shape[-1] == 6:
        x, z = obs[:, 0], obs[:, 2]
        x_goal = torch.as_tensor(info["goal"][0])
        z_goal = torch.as_tensor(info["goal"][1])
        cost = (x - x_goal)**2 + (z - z_goal)**2
    else:
        raise NotImplementedError
    # cost += 0.01 * torch.sum(torch.square(act), -1)
    return cost


def quadrotor_cost_info_func(info, env):
    """Per-step function to augment info dict for cost func.
    """
    info["goal"] = env.TASK_INFO["stabilization_goal"]
    return info


ENV_COST_FUNCS = {
    "cartpole": {
        "cost_func": cartpole_cost_func,
        "cost_info_func": cartpole_cost_info_func,
    },
    "quadrotor": {
        "cost_func": quadrotor_cost_func,
        "cost_info_func": quadrotor_cost_info_func,
    },
}
