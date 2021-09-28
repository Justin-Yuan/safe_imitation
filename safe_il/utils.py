import random
import numpy as np
import os
import sys
import torch
import yaml


def set_manual_seed(seed):
    """
    To create reproducible results, set all seeds across all RNG manually,
    https://github.com/pytorch/pytorch/issues/7068#issuecomment-484918113
    when using additional workers, those also need to set their seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    # TODO:(Mustafa) This should be True for reproducilbility, but letting
    # it benchmark increases speed dramatically, so swap this for final runs
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def save_checkpoint(log_dir, model_name, checkpoint_dict):
    """Save the full pytorch model checkpoint."""
    file_path = os.path.join(log_dir, model_name + '.pth')
    torch.save(checkpoint_dict, file_path)


def save_config(config, output_dir):
    """Logs configs to file under directory."""
    config_dict = config.__dict__
    file_path = os.path.join(output_dir, "config.yaml")
    with open(file_path, "w") as file:
        yaml.dump(config_dict, file, default_flow_style=False)


def save_command(output_dir):
    """Logs current executing command to text file."""
    with open(os.path.join(output_dir, 'cmd.txt'), 'a') as file:
        file.write(" ".join(sys.argv) + "\n")


def random_sample(indices, batch_size, drop_last=True):
    """Returns index batches to iterave over"""
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    if not drop_last:
        r = len(indices) % batch_size
        if r:
            yield indices[-r:]


def get_random_state():
    """Snapshots the random state at any moment."""
    return {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state()
    }


def set_random_state(state_dict):
    """Resets the random state for experiment restore."""
    random.setstate(state_dict["random"])
    np.random.set_state(state_dict["numpy"])
    torch.torch.set_rng_state(state_dict["torch"])
    

def unwrap_wrapper(env, wrapper_class):
    """Retrieve a ``VecEnvWrapper`` object by recursively searching.

    Reference:
        * https://github.com/DLR-RM/stable-baselines3/blob/ddbe0e93f9fe55152f2354afd058b28e6ccc3345/stable_baselines3/common/env_util.py
    """
    env_tmp = env
    while isinstance(env_tmp, gym.Wrapper):
        if isinstance(env_tmp, wrapper_class):
            return env_tmp
        env_tmp = env_tmp.env
    return None


def is_wrapped(env, wrapper_class):
    """Check if a given environment has been wrapped with a given wrapper.

    Reference:
        * https://github.com/DLR-RM/stable-baselines3/blob/ddbe0e93f9fe55152f2354afd058b28e6ccc3345/stable_baselines3/common/env_util.py
    """
    return unwrap_wrapper(env, wrapper_class) is not None