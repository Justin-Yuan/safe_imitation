"""Vanilla training script for model-based rl methods (mpc, pets).

Example: 
    To train PETS on cartpole:: 

        $ python scripts/train_mbrl.py --algo pets --env cartpole --config configs/mbrl/pets_cartpole.yaml --tag pets_cartpole --device cuda --seed 1

    To test trained model::
    
        $ python scripts/train_mbrl.py --func test --restore <path to trained folder>
    
Todo:
    *

"""
import os
import sys
# TODO: hack
sys.path.insert(0, os.getcwd())
import pickle
import torch

from safe_il.agents import AGENTS
from safe_il.envs import ENVS
from safe_il.config2 import ConfigFactory
from safe_il.utils import (
    mkdirs, set_dir_from_config, set_device_from_config, 
    set_seed_from_config, save_video
)


# -----------------------------------------------------------------------------------
#                   Funcs
# -----------------------------------------------------------------------------------


def train(config):
    """General training template.
    
    Usage:
        * to start training, use with `--func train`.
        * to restore from a previous training, additionally use `--restore {dir_path}` 
            where `dir_path` is the output folder from previous training.  
    """
    # Experiment setup
    if not config.restore:
        set_dir_from_config(config)
    set_seed_from_config(config)
    set_device_from_config(config)

    # Create env
    env_func = ENVS[config.env]
    env = env_func(**config.env_config)
    env.seed(config.seed)
    eval_env = env_func(**config.env_config)
    eval_env.seed(config.seed * 111 if config.seed is not None else None)

    # Create agent
    agent_func = AGENTS[config.algo]
    agent = agent_func(
        env, 
        eval_env=eval_env,
        training=True,
        checkpoint_path=os.path.join(config.output_dir, "model_latest.pt"),
        output_dir=config.output_dir,
        device=config.device,
        seed=config.seed,
        **config.algo_config
    )
    agent.reset()
    if config.restore:
        agent.load(os.path.join(config.restore, "model_latest.pt"))

    # Training
    agent.learn()
    agent.close()
    print("Training done.")


def test(config):
    """Run the (trained) policy/controller for evaluation.
    
    Usage
        * use with `--func test`.
        * to test policy from a trained model checkpoint, additionally use 
            `--restore {dir_path}` where `dir_path` is folder to the trained model.
        * to test un-trained policy (e.g. non-learning based), use as it is.
    """
    # Evaluation setup
    if config.set_test_seed:
        set_seed_from_config(config)
    set_device_from_config(config)

    # Create env
    env_func = ENVS[config.env]
    env = env_func(**config.env_config)
    if config.set_test_seed:
        env.seed(config.seed)

    # Create agent
    agent_func = AGENTS[config.algo]
    agent = agent_func(
        env,
        training=False,
        checkpoint_path=os.path.join(config.output_dir, "model_latest.pt"),
        output_dir=config.output_dir,
        device=config.device,
        seed=config.seed,
        **config.algo_config
    )
    agent.reset()
    if config.restore:
        agent.load(os.path.join(config.restore, "model_latest.pt"))

    # Test agent
    results = agent.run(n_episodes=config.n_episodes, render=config.render, verbose=config.verbose)
    agent.close()

    # Save evalution results
    eval_path = os.path.join(config.output_dir, "eval", config.eval_output_path)
    eval_dir = os.path.dirname(eval_path)
    mkdirs(eval_dir)
    with open(eval_path, "wb") as f:
        pickle.dump(results, f)

    ep_lengths = results["ep_lengths"]
    ep_returns = results["ep_returns"]
    msg = "eval_ep_length {:.2f} +/- {:.2f}\n".format(ep_lengths.mean(), ep_lengths.std())
    msg += "eval_ep_return {:.3f} +/- {:.3f}\n".format(ep_returns.mean(), ep_returns.std())
    print(msg)

    if "frames" in results:
        save_video(os.path.join(eval_dir, "video.gif"), results["frames"])
    print("Evaluation done.")
    
    
# -----------------------------------------------------------------------------------
#                   Main
# -----------------------------------------------------------------------------------

MAIN_FUNCS = {"train": train, "test": test}

if __name__ == "__main__":
    # Make config
    fac = ConfigFactory()
    fac.add_argument("--func", type=str, default="train", help="main function to run.")
    fac.add_argument("--thread", type=int, default=0, help="number of threads to use (set by torch).")
    fac.add_argument("--render", action="store_true", help="if to render in policy test.")
    fac.add_argument("--verbose", action="store_true", help="if to print states & actions in policy test.")
    fac.add_argument("--eval_output_path", type=str, default="test_results.pkl", help="file path to save evaluation results.")
    fac.add_argument("--set_test_seed", action="store_true", help="if to set seed when testing policy.")
    fac.add_argument("--n_episodes", type=int, default=10, help="number of test episodes.")
    config = fac.merge()

    # system settings
    if config.thread > 0:
        # e.g. set single thread for less context switching
        torch.set_num_threads(config.thread)

    # Execute
    func = MAIN_FUNCS.get(config.func, None)
    if func is None:
        raise Exception("Main function {} not supported.".format(config.func))
    func(config)
