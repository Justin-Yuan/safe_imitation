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
