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


# Get activation from PyTorch Functional
def get_activation(name):
    return getattr(F, name) if name else lambda x: x


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class QSafeNetwork(nn.Module):
    '''
    Safety-Q network

    Arguments:
        input_dim (int): input dimension
        action_dim (int): action space dimension
        hidden_dim (int): hidden layer dimension
        act (str): activation function to use
        output_act (str): activation function for output
        num_layers (int): number of hidden layers
        use_dropout (bool): enable dropout
        init_weights (bool): enable initialization of weights using Xavier
    '''
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim,
                 act='leaky_relu',
                 output_act=None,
                 num_layers=2,
                 use_dropout=False,
                 init_weights=True):

        super(QSafeNetwork, self).__init__()

        dims = [input_dim] + ([hidden_dim] * num_layers) + [output_dim]
        self.linears = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        self.use_dropout = use_dropout
        self.act = get_activation(act)
        self.output_act = get_activation(act)
        if init_weights:
            self.apply(weights_init_)

    def forward(self, input):

        out = input

        for lin in self.linears[:-1]:
            out = self.act(lin(out))

        if self.use_dropout:
            out = F.dropout(out)

        out = self.output_act(self.linears[-1](out))

        return out
