import os
import time
import copy
from collections import defaultdict
from functools import partial
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from munch import munchify

from safe_il.utils import set_manual_seed
from safe_il.pets.mpc_utils import MPCAgent


class MPC
    """CEM-based MPC."""

    def __init__(
        self, 
        env, 
        eval_env=None,
        training=True, 
        checkpoint_path="model_latest.pt", 
        output_dir="temp", 
        device="cpu", 
        seed=None, 
        **kwargs
    ):
        # common args 
        self.training = training
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        self.device = device
        self.seed = seed 
        
        # task
        self.env = env
        self.eval_env = eval_env 

        # agent
        self.agent = MPCAgent(self.env.observation_space,
                              self.env.action_space,
                              self.env.cost,
                              self.env.model,
                              horizon=self.horizon,
                              cem_args=self.cem)
        self.agent.to(device)

    def reset(self):
        """Prepares for training or evaluation."""
        pass

    def close(self):
        """Shuts down and cleans up lingering resources."""
        self.env.close()
        if self.eval_env is not None:
            self.eval_env.close()

    def run(self, env=None, render=False, n_episodes=10, verbose=False, **kwargs):
        """Runs evaluation with current policy."""
        self.agent.eval()
        if env is None:
            env = self.env

        ep_returns, ep_lengths = [], []
        frames = []
        
        t = 0
        ep_length, ep_return = 0, 0 
        self.agent.reset()
        obs, info = env.reset()
        # obs = self.obs_normalizer(obs)

        while len(ep_returns) < n_episodes:
            with torch.no_grad():
                action = self.agent.act(obs, t, info)

            obs, reward, done, info = env.step(action)
            t += 1
            ep_length += 1 
            ep_return += reward 
            if render:
                env.render()
                frames.append(env.render("rgb_array"))
            if verbose:
                print("t {} | obs {} | act {}".format(t, obs, action))

            if done:
                ep_lengths.append(ep_length)
                ep_returns.append(ep_return)
                
                t = 0
                ep_length, ep_return = 0, 0 
                self.agent.reset()
                obs, info = env.reset()
            # obs = self.obs_normalizer(obs)

        # collect evaluation results
        ep_lengths = np.asarray(ep_lengths)
        ep_returns = np.asarray(ep_returns)
        eval_results = {"ep_returns": ep_returns, "ep_lengths": ep_lengths}
        if len(frames) > 0:
            eval_results["frames"] = frames
        return eval_results


# -----------------------------------------------------------------------------------
#                   Tests
# -----------------------------------------------------------------------------------


def test_mpc_cartpole():
    """Run the (trained) policy/controller for evaluation.
    """
    from safe_il.envs.cartpole import CartPole

    config = {
        "seed": 1234,
        "task_config": {
            "normalized_action": True,
        }, 
        "algo_config": {
            "horizon": 25,
            "cem": {
                "pop_size": 400,
                "num_elites": 40,
                "max_iters": 5,
                "alpha": 0.1,
            },
            "deque_size": 10,
        },  
    }
    config = munchify(config)
    set_manual_seed(config.seed)
    
    # Create task/env
    env = CartPole(**config.task_config)
    env.seed(config.seed)

    # Create the controller/control_agent.
    agent = MPC(env, seed=config.seed, device="cuda", **config.algo_config)
    agent.reset()

    # Test controller
    results = agent.run(render=False, verbose=True)
    agent.close()

    # Save evalution results
    ep_lengths = results["ep_lengths"]
    ep_returns = results["ep_returns"]
    msg = "eval_ep_length {:.2f} +/- {:.2f}\n".format(ep_lengths.mean(), ep_lengths.std())
    msg += "eval_ep_return {:.3f} +/- {:.3f}\n".format(ep_returns.mean(), ep_returns.std())
    print(msg)
    print("Evaluation done.")


def test_cartpole_dynamics_deviation():
    """Difference between analytical dynamics model and ground truth model.
    """
    import matplotlib.pyplot as plt
    from safe_il.envs.cartpole import CartPole

    config = {
        "seed": 1234,
        "task_config": {
            "normalized_action": True,
        }, 
    }
    config = munchify(config)
    
    env = CartPole(**config.task_config)
    env.seed(config.seed)

    obs, info = env.reset()
    obs2 = torch.as_tensor(obs)
    model = env.model

    diffs = []
    test_steps = 1000
    for i in range(test_steps):
        action = env.action_space.sample() * 0
        obs, _, done, info = env.step(action)
        obs2 = model(obs2, torch.as_tensor(action))

        diffs.append(obs - obs2.numpy())
        if done:
            break
    env.close()

    x = list(range(len(diffs)))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.plot(x, [float(d[i]) for d in diffs])

    plt.savefig("figures/test_cartpole_dynamics_deviation.png")
    plt.show()


if __name__ == "__main__":
    test_mpc_cartpole()
    # test_cartpole_dynamics_deviation()