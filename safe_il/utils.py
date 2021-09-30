import random
import numpy as np
import os
import sys
import torch
import yaml
import datetime
import gym
import json
import subprocess
import munch
import imageio



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


def mkdirs(*paths):
    """Makes a list of directories."""
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def eval_token(token):
    """Converts string token to int, float or str."""
    if token.isnumeric():
        return int(token)
    try:
        return float(token)
    except TypeError:
        return token


def read_file(file_path, sep=","):
    """Loads content from a file (json, yaml, csv, txt).
    
    For json & yaml files returns a dict.
    Ror csv & txt returns list of lines.
    """
    if len(file_path) < 1 or not os.path.exists(file_path):
        return None
    # load file
    f = open(file_path, "r")
    if "json" in file_path:
        data = json.load(f)
    elif "yaml" in file_path:
        data = yaml.load(f, Loader=yaml.FullLoader)
    else:
        sep = sep if "csv" in file_path else " "
        data = []
        for line in f.readlines():
            line_post = [eval_token(t) for t in line.strip().split(sep)]
            # if only sinlge item in line
            if len(line_post) == 1:
                line_post = line_post[0]
            if len(line_post) > 0:
                data.append(line_post)
    f.close()
    return data


def merge_dict(source_dict, update_dict):
    """Merges updates into source recursively."""
    for k, v in update_dict.items():
        if k in source_dict and isinstance(source_dict[k], dict) and isinstance(v, dict):
            merge_dict(source_dict[k], v)
        else:
            source_dict[k] = v
            

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
    
    
def set_dir_from_config(config):
    """Creates a output folder for experiment (and save config files).
    
    Naming format: {root (e.g. results)}/{tag (exp id)}/{seed}_{timestamp}_{git commit id}
    """
    # Make output folder.
    timestamp = datetime.datetime.now().strftime("%b-%d-%H-%M-%S")
    commit_id = subprocess.check_output(
        ["git", "describe", "--tags", "--always"]).decode("utf-8").strip()
    run_dir = "seed{}_{}_{}".format(str(config.seed) if config.seed is not None else "",
                                    str(timestamp),
                                    str(commit_id)
                                    )
    config.output_dir = os.path.join(config.output_dir, config.tag, run_dir)
    mkdirs(config.output_dir)
    # Save config.
    with open(os.path.join(config.output_dir, 'config.yaml'), "w") as file:
        yaml.dump(munch.unmunchify(config), file, default_flow_style=False)
    # Save command.
    with open(os.path.join(config.output_dir, 'cmd.txt'), 'a') as file:
        file.write(" ".join(sys.argv) + "\n")


def set_seed_from_config(config):
    """Sets seed only if provided."""
    if config.seed is not None:
        set_manual_seed(config.seed)


def set_device_from_config(config):
    """Sets device, using GPU is set to `cuda` for now, no specific GPU yet."""
    use_cuda = (config.device == "cuda") and torch.cuda.is_available()
    config.device = "cuda" if use_cuda else "cpu"


def save_video(name, frames, fps=20):
    """Convert list of frames (H,W,C) to a video.

    Args:
        name (str): path name to save the video.
        frames (list): frames of the video as list of np.arrays.
        fps (int, optional): frames per second.

    """
    assert ".gif" in name or ".mp4" in name, "invalid video name"
    vid_kwargs = {'fps': fps}
    h, w, c = frames[0].shape
    video = np.stack(frames, 0).astype(np.uint8).reshape(-1, h, w, c)
    imageio.mimsave(name, video, **vid_kwargs)
    

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