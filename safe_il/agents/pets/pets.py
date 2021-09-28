"""Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models

URL: https://arxiv.org/pdf/1805.12114.pdf

References: 
    * (tf) https://github.com/kchua/handful-of-trials
    * (torch) https://github.com/quanvuong/handful-of-trials-pytorch 
 
Note: specify either `init_rollouts` or `init_steps` but not both,
the sanme with `rollouts_per_iter` and `steps_per_iter`. Original algo
uses rollouts but some task/env can terminate early (with only a few steps), 
so if you need to force update on a minimum number of steps specify steps instead.

"""
import os
import time
import copy
from collections import defaultdict
from functools import partial
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from safe_il.logging import ExperimentLogger
from safe_il.utils import get_random_state, set_random_state, is_wrapped
from safe_il.envs.record_episode_statistics import RecordEpisodeStatistics
from safe_il.agents.pets.pets_utils import PETSAgent, PETSBuffer


# -----------------------------------------------------------------------------------
#                   Algo
# -----------------------------------------------------------------------------------


class PETS:
    """Probabilistic Ensembles with Trajectory Sampling."""

    def __init__(
        self,
        env, 
        eval_env=None,
        training=True, 
        checkpoint_path="model_latest.pt", 
        output_dir="temp", 
        device="cpu", 
        seed=None, 
        # custom args
        hidden_dim=500,
        ensemble_size=5,
        weight_decays=[1.e-4, 2.5e-4, 2.5e-4, 5.e-4],
        lr=0.001,
        epochs=5,
        batch_size=256,
        horizon=25,
        num_particles=20,
        cem={
            "pop_size": 400,
            "num_elites": 40,
            "max_iters": 5,
            "alpha": 0.1  
        },
        # runner args
        init_rollouts=1,
        init_steps=0,
        train_iters=50,
        rollouts_per_iter=1,
        steps_per_iter=0,
        deque_size=10,
        eval_batch_size=10,
        # misc
        log_interval=0,
        save_interval=0,
        num_checkpoints=0,
        eval_interval=0,
        eval_save_best=False,
        tensorboard=False,
        **kwargs
    ):
        # bookkeeping  
        self.training = training
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        self.device = device
        self.seed = seed 
        
        self.hidden_dim = hidden_dim
        self.weight_decays = weight_decays
        self.ensemble_size = ensemble_size
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.horizon = horizon
        self.num_particles = num_particles
        self.cem = cem
        
        self.init_rollouts = init_rollouts
        self.init_steps = init_steps
        self.train_iters = train_iters
        self.rollouts_per_iter = rollouts_per_iter
        self.steps_per_iter = steps_per_iter
        self.deque_size = deque_size
        self.eval_batch_size = eval_batch_size
        
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.num_checkpoints = num_checkpoints
        self.eval_interval = eval_interval
        self.eval_save_best = eval_save_best
        self.tensorboard = tensorboard
        
        # task
        self.env = RecordEpisodeStatistics(env, self.deque_size)
        if self.training and eval_env is not None:
            self.eval_env = RecordEpisodeStatistics(eval_env, self.deque_size) 

        # agent
        self.agent = PETSAgent(self.env.observation_space,
                               self.env.action_space,
                               self.env.cost,
                               hidden_dim=self.hidden_dim,
                               ensemble_size=self.ensemble_size,
                               weight_decays=self.weight_decays,
                               lr=self.lr,
                               epochs=self.epochs,
                               batch_size=self.batch_size,
                               horizon=self.horizon,
                               num_particles=self.num_particles,
                               cem_args=self.cem)
        self.agent.to(device)

        # logging
        if self.training:
            log_file_out = True
            use_tensorboard = self.tensorboard
        else:
            # disable logging to texts and tfboard for testing
            log_file_out = False
            use_tensorboard = False
        self.logger = ExperimentLogger(output_dir, log_file_out=log_file_out, use_tensorboard=use_tensorboard)

    def reset(self):
        """Prepares for training or evaluation."""
        if self.training:
            self.total_iters = 0
            self.buffer = PETSBuffer(self.env.observation_space, self.env.action_space)

    def close(self):
        """Shuts down and cleans up lingering resources."""
        self.env.close()
        if self.training:
            self.eval_env.close()
        self.logger.close()

    def save(self, path):
        """Saves model params and experiment state to checkpoint path."""
        path_dir = os.path.dirname(path)
        os.makedirs(path_dir, exist_ok=True)

        state_dict = {
            "agent": self.agent.state_dict(),
        }
        if self.training:
            exp_state = {
                "total_iters": self.total_iters,
                "random_state": get_random_state(),
                "buffer": self.buffer.state_dict(),
            }
            state_dict.update(exp_state)
        torch.save(state_dict, path)

    def load(self, path):
        """Restores model and experiment given checkpoint path."""
        state = torch.load(path)

        # restore policy
        self.agent.load_state_dict(state["agent"])

        # restore experiment state
        if self.training:
            self.total_iters = state["total_iters"]
            set_random_state(state["random_state"])
            self.buffer.load_state_dict(state["buffer"])
            self.logger.load(self.total_iters)

    def learn(self, env=None, **kwargs):
        """Performs learning (pre-training, training, fine-tuning, etc)."""
        # initial train step
        if self.init_rollouts > 0 or self.init_steps > 0:
            # collect initial data (with random controller)
            rollouts = self.collect_rollouts(num_rollouts=self.init_rollouts, num_steps=self.init_steps, explore=True)
            input_batch, target_batch = self.process_rollouts(rollouts)
            self.buffer.push(input_batch, target_batch)
            # train
            self.agent.train()
            self.agent.update(self.buffer, device=self.device)

        while self.total_iters < self.train_iters:
            results = self.train_step()

            # checkpoint
            if self.total_iters >= self.train_iters or (self.save_interval and self.total_iters % self.save_interval == 0):
                # latest/final checkpoint
                self.save(self.checkpoint_path)
                self.logger.info("Checkpoint | {}".format(self.checkpoint_path))
            if self.num_checkpoints and self.total_iters % (self.train_iters // self.num_checkpoints) == 0:
                # intermediate checkpoint
                path = os.path.join(self.output_dir, "checkpoints", "model_{}.pt".format(self.total_iters))
                self.save(path)

            # eval
            if self.eval_interval and self.total_iters % self.eval_interval == 0:
                eval_results = self.run(env=self.eval_env, n_episodes=self.eval_batch_size)
                results["eval"] = eval_results
                self.logger.info("Eval | ep_lengths {:.2f} +/- {:.2f} | ep_return {:.3f} +/- {:.3f}".format(eval_results["ep_lengths"].mean(),
                                                                                                            eval_results["ep_lengths"].std(),
                                                                                                            eval_results["ep_returns"].mean(),
                                                                                                            eval_results["ep_returns"].std()))
                # save best model
                eval_score = eval_results["ep_returns"].mean()
                eval_best_score = getattr(self, "eval_best_score", -np.infty)
                if self.eval_save_best and eval_best_score < eval_score:
                    self.eval_best_score = eval_score
                    self.save(os.path.join(self.output_dir, "model_best.pt"))

            # logging
            if self.log_interval and self.total_iters % self.log_interval == 0:
                self.log_step(results)

    def run(self, env=None, render=False, n_episodes=10, verbose=False, **kwargs):
        """Runs evaluation with current policy."""
        self.agent.eval()
        if env is None:
            env = self.env
        else:
            if not is_wrapped(env, RecordEpisodeStatistics):
                env = RecordEpisodeStatistics(env)

        ep_returns, ep_lengths = [], []
        frames = []
        
        t = 0
        self.agent.reset()
        obs, info = env.reset()
        # obs = self.obs_normalizer(obs)

        while len(ep_returns) < n_episodes:
            with torch.no_grad():
                action = self.agent.act(obs, t, info)

            obs, reward, done, info = env.step(action)
            t += 1
            if render:
                env.render()
                frames.append(env.render("rgb_array"))
            if verbose:
                print("t {} | obs {} | act {}".format(t, obs, action))

            if done:
                assert "episode" in info
                ep_returns.append(info["episode"]["r"])
                ep_lengths.append(info["episode"]["l"])

                t = 0
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

    def train_step(self, **kwargs):
        """Performs a pre-trianing or adaptation step."""
        self.agent.train()
        start = time.time()

        # collect data
        rollouts = self.collect_rollouts(num_rollouts=self.rollouts_per_iter, num_steps=self.steps_per_iter)
        input_batch, target_batch = self.process_rollouts(rollouts)
        self.buffer.push(input_batch, target_batch)
        self.total_iters += 1

        # learn
        results = self.agent.update(self.buffer, device=self.device)
        results.update({"iter": self.total_iters, "elapsed_time": time.time() - start})
        return results

    def log_step(self, results):
        """Does logging after a training step."""
        n_iter = results["iter"]
        # runner stats
        self.logger.add_scalars(
            {
                "iter": n_iter,
                "iter_time": results["elapsed_time"],
                "progress": n_iter / self.train_iters
            },
            n_iter,
            prefix="time",
            write=False,
            write_tb=False
        )

        # learning stats
        self.logger.add_scalars({k: results[k] for k in ["nll_loss", "mse_loss", "reg_loss"]}, n_iter, prefix="loss")

        # performance stats
        ep_lengths = np.asarray(self.env.length_queue)
        ep_returns = np.asarray(self.env.return_queue)
        self.logger.add_scalars(
            {
                "ep_length": ep_lengths.mean(),
                "ep_return": ep_returns.mean(),
                "ep_reward": (ep_returns / ep_lengths).mean()
            },
            n_iter,
            prefix="stat"
        )

        if "eval" in results:
            eval_ep_lengths = results["eval"]["ep_lengths"]
            eval_ep_returns = results["eval"]["ep_returns"]
            self.logger.add_scalars(
                {
                    "ep_length": eval_ep_lengths.mean(),
                    "ep_return": eval_ep_returns.mean(),
                    "ep_reward": (eval_ep_returns / eval_ep_lengths).mean()
                },
                n_iter,
                prefix="stat_eval"
            )

        # print summary table
        self.logger.dump_scalars()

    def collect_rollouts(self, num_rollouts=0, num_steps=0, explore=False):
        """Samples rollouts with the agent.
        
        Args:
            num_rollouts (int): number of rollouts to collect.
            num_steps (int): number of minimum steps to collect, actual number 
                might exceed since full rollouts are collected.
            explore (bool): if to explore with random actions or use planning.

        Returns: 
            list: list of dicts containing data from the rollouts.
        """
        # only one of them should be used
        assert (num_rollouts > 0) ^ (num_steps > 0)

        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        rollouts = []
        n_rollouts = 0
        n_steps = 0

        while True:
            obs, info = self.env.reset()
            obs = obs.reshape(obs_dim)
            self.agent.reset()

            ret, done, t = 0, False, 0
            traj_obs, traj_acts, traj_rews = [copy.deepcopy(obs)], [], []

            while not done:
                if explore:
                    action = self.env.action_space.sample()
                else:
                    cost_info = self.env_cost_info_func(info, self.env)
                    with torch.no_grad():
                        action = self.agent.act(obs, t, cost_info)
                action = action.reshape(act_dim)

                obs, rew, done, info = self.env.step(action)
                obs = obs.reshape(obs_dim)

                traj_obs.append(copy.deepcopy(obs))
                traj_acts.append(copy.deepcopy(action))
                traj_rews.append(copy.deepcopy(rew))
                ret += rew
                t += 1
                n_steps += 1

            # collect each rollout stats
            rollouts.append({
                "obs": np.stack(traj_obs, 0),
                "act": np.stack(traj_acts, 0),
                "return": ret,
                "rewards": np.stack(traj_rews, 0),
            })
            n_rollouts += 1

            # check if data is enough
            if num_rollouts > 0 and n_rollouts >= num_rollouts:
                break
            if num_steps > 0 and n_steps >= num_steps:
                break

        return rollouts

    def process_rollouts(self, rollouts):
        """Convert raw rollouts to training data for model.
        
        Returns:
            tuple of lists: processed input and target data.
        """
        input_batch, target_batch = [], []

        for rollout in rollouts:
            # each is (T,*) or (T+1,*) for obs
            obs, acts = rollout["obs"], rollout["act"]

            # input for model is (obs,act) pair at each step
            inputs = np.concatenate([obs[:-1], acts], -1)
            # target is state residual
            targets = obs[1:] - obs[:-1]

            input_batch.append(inputs)
            target_batch.append(targets)

        # each is [(T,*)]_{num_rollouts}
        return input_batch, target_batch


# -----------------------------------------------------------------------------------
#                   Tests
# -----------------------------------------------------------------------------------


def test_pets_cartpole():
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