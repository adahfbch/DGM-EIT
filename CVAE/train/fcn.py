import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, input=208, z_dim=16):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input, 256),
            nn.LeakyReLU(),

            nn.Linear(256, 128),
            nn.LeakyReLU(),

            nn.Linear(128, 64),
            nn.LeakyReLU(),

            nn.Linear(64, 32),
            nn.LeakyReLU(),

            nn.Linear(32, z_dim),
            nn.Dropout(),
            nn.Tanh()
        )


    def forward(self, x):
        x = self.model(x)
        return x*5  # 5



def loss_fz(recon_z, z, batch_size):
    MSE = torch.nn.MSELoss(reduction='sum')(z, recon_z)/batch_size

    return MSE