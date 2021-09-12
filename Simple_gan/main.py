import torch
import argparse

import torchvision.utils

from Model import Generator, Discriminator
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from torchvision.transforms import Compose
from torchvision import transforms
# Note: you can use any other dataset you want,
# as long as you change the transforms appropriately
from torchvision.datasets import MNIST
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate. Default: 2e-4")
parser.add_argument("--niter", type=int, default=25)

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

lr = args.lr
epochs = args.niter

noise_dim = 100
batch_size = 32
img_dim = 28**2
n_colors = 1
height = 28
width = 28


ref_noise = torch.randn((batch_size, noise_dim)).to(device)
disc = Discriminator(img_dim).to(device)
gen = Generator(noise_dim, img_dim).to(device)
transformations = Compose([
        transforms.ToTensor(),
        transforms.Normalize(.5, .5),
    ])
path = os.path.join(os.getcwd(), "dataset")
ds = MNIST(path, transform=transformations, download=True)

loader = DataLoader(ds, batch_size=batch_size, shuffle=True)


disc_opt = optim.Adam(disc.parameters(), lr=lr)
gen_opt = optim.Adam(gen.parameters(), lr=lr)
loss = nn.BCELoss()
# min_G max_D log(D(x)) + log(1 - D(G(y))
# Equivalent to :
real_writer_path = os.path.join(os.getcwd(), "logs", "real")
fake_writer_path = os.path.join(os.getcwd(), "logs", "fake")
write_real = SummaryWriter(real_writer_path)
write_fake = SummaryWriter(fake_writer_path)

bar = tqdm(range(epochs))
for epoch in bar:
    for num, (real, _) in enumerate(loader):
        real = real.view(-1, img_dim).to(device)
        noise = torch.randn(batch_size, noise_dim).to(device)
        # Generate a fake image

        fake = gen(noise)
        # Look at what the discriminator says
        disc_out_real = disc(real).view(-1)
        disc_out_fake = disc(fake).view(-1)

        # BCE(x,y) = - (y log(x) + (1 - y) log(1 - x))

        # disc -> max log(D(x)) + log(1 - D(G(y))
        # equivalent to min - (log(D(x)) + log(1 - D(G(y)))

        # discriminator loss real -> log(D(x))
        # with BCE : discriminator loss real =
        disc_loss_real = loss(disc_out_real, torch.ones_like(disc_out_real))
        disc_loss_fake = loss(disc_out_fake, torch.ones_like(disc_out_fake))

        disc_loss = disc_loss_real + disc_loss_fake

        disc.zero_grad()
        disc_loss.backward(retain_graph=True)

        disc_opt.step()

        # let's do the generator now:
        # Note: min_G log(1 - D(G(y)) leads to saturated gradients
        # we instead take max_G log(D(G(y))
        # or min_G - log(D(G(y))

        out_disc_fake = disc(fake).view(-1)
        gen_loss = loss(out_disc_fake, torch.ones_like(out_disc_fake))
        gen.zero_grad()
        gen_loss.backward()
        gen_opt.step()

    bar.set_description(f"loss disc: {disc_loss}, loss gen {gen_loss}")
    with torch.no_grad():
        fake = gen(ref_noise).view(-1, n_colors, height, width)
        real_view = real.view(-1, n_colors, height, width)
        fake_grid = torchvision.utils.make_grid(fake, normalize=True)
        real_grid = torchvision.utils.make_grid(real_view, normalize=True)

        write_fake.add_image("Fake", fake_grid, global_step=epoch)
        write_real.add_image("Real", real_grid, global_step=epoch)
