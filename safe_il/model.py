'''
Models built from implementations found in RecoveryRL
https://github.com/abalakrishna123/recovery-rl/, T-Rex implementation at 
https://github.com/Stanford-ILIAD/TREX-pytorch/blob/master/models.py, and 
Latent dynamics models are built on latent dynamics model used in
Goal-Aware Prediction: Learning to Model What Matters (ICML 2020). All
other networks are built on SAC implementation from
https://github.com/pranz24/pytorch-soft-actor-critic
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class QSafeNetwork(nn.Module):
    '''
    Safety-Q network
    '''
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QSafeNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        inp = torch.cat([state, action], 1)

        out = F.relu(self.linear1(inp))
        out = F.relu(self.linear2(out))
        out = self.linear3(out)

        return out
