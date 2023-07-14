from torch import nn

import torch


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=8192):
        return input.view(input.size(0), size, 1, 1)


class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=8192, z_dim=16):  # h_dim 确定到底咋算的，我算的是C*H*W,但是报错算出来的h_dim和我算的不太一致
        super(VAE, self).__init__()
        self.latent_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 16, kernel_size=5, stride=2, padding=2),
            # nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),


            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),



            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),


            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),

            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Sequential(nn.Linear(z_dim, h_dim), UnFlatten())

        self.decoder = nn.Sequential(
            # UnFlatten(),
            # nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=8, padding=2, output_padding=1),
            # nn.ReLU(),
            # nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1,bias=False),  # ,
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1,bias=False),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.BatchNorm2d(32),


            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1,bias=False),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.ConvTranspose2d(16, image_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU()
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = z.view(z.shape[0], 128, 8, 8)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

    def sample(self,
               num_samples:int,**kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the pymodel
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        samples = self.decode(z)
        return samples


def loss_fn(recon_x, x, mu, logvar, batch_size):
    MSE = torch.nn.MSELoss(reduction='sum')(x, recon_x)/batch_size
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())/batch_size

    return MSE + KLD, MSE, KLD

