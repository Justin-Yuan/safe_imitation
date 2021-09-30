import numpy as np
import torch
from torch.utils.data import Dataset


def generate_trajectory_data():
    return NotImplementedError


class PartialTrajectoryDataset(Dataset):
    """
    This dataset returns a pair of partial trajectories, and the sum of
    their safety costs
    """
    def __init__(self, path):
        self.path = path
        self.trajectories = np.load(path, allow_pickle=True)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):

        rand_idx = torch.randint(0, len(self.trajectories) - 1, (1,))
        t1 = self.trajectories[idx]
        t2 = self.trajectories[rand_idx]

        t1_inputs = torch.Tensor(
            [traj[0] for traj in t1])
        t2_inputs = torch.Tensor(
            [traj[0] for traj in t2])

        q1 = torch.sum(torch.Tensor([traj[3] for traj in t1]))
        q2 = torch.sum(torch.Tensor([traj[3] for traj in t2]))

        return t1_inputs, t2_inputs, q1, q2
