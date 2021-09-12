from torch import nn


class Generator(nn.Module):
    def __init__(self, noise_dim, img_dim):
        self.gen = nn.Sequential(
            nn.Linear(noise_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, img_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.gen(x)


class Discriminator(nn.Module):
    def __init__(self, img_dim):
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)
