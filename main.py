import argparse
import torch
from torch.optim import Adam
from torchvision.utils import save_image
from torch.nn import Upsample
from network import Generator, Discriminator
from data_loader import get_loader
import time
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

upsample = []

def print_debugging_images(generator, latent_vectors, shape, index, alpha,
                           iteration):
    global upsample
    with torch.no_grad():
        columns = []
        for i in range(shape[0]):
            row = []
            for j in range(shape[1]):
                img_ij = generator(latent_vectors[i*shape[1] +
                                                  j].unsqueeze_(0),
                                   index, alpha)
                img_ij = upsample[index](img_ij)
                row.append(img_ij)
            columns.append(torch.cat(row, dim=3))
        debugging_image = torch.cat(columns, dim=2)
    # denorm
    debugging_image = (debugging_image + 1) / 2
    debugging_image.clamp_(0, 1)
    save_image(debugging_image.data, "img/debug_{}_{}.png".format(index,
                                                               iteration))


def train(data_path, crop_size=128, final_size=64, batch_size=16,
          alternating_step=10000, ncritic=1, lambda_gp=0.1, debug_step=100):
    # define networks
    generator = Generator(final_size=final_size)
    generator.generate_network()
    g_optimizer = Adam(generator.parameters())

    discriminator = Discriminator(final_size=final_size)
    discriminator.generate_network()
    d_optimizer = Adam(discriminator.parameters())

    num_channels = min(generator.num_channels, generator.max_channels)

    # get debugging vectors
    N = (5, 10)
    debug_vectors = torch.randn(N[0]*N[1], num_channels, 1, 1).to(device)
    global upsample
    upsample = [Upsample(scale_factor=2**i)
                for i in reversed(range(generator.num_blocks))]

    # get loader
    loader = get_loader(data_path, crop_size, batch_size)

    # training loop
    start_time = time.time()
    for index in range(generator.num_blocks):
        loader.dataset.set_transform_by_index(index)
        data_iterator = iter(loader)
        for phase in ('fade', 'stabilize'):
            if index == 0 and phase == 'fade': continue
            print("index: {}, size: {}x{}, phase: {}".format(index,
                                                             2**(index+2),
                                                             2**(index+2),
                                                             phase))
            for i in range(alternating_step):
                print(i)
                try:
                    batch = next(data_iterator)
                except:
                    data_iterator = iter(loader)
                    batch = next(data_iterator)

                alpha = i / alternating_step if phase == "fade" else 1.0

                batch = batch.to(device)

                d_loss_real = - torch.mean(discriminator(batch, index, alpha))

                latent = torch.randn(batch_size, num_channels, 1, 1).to(device)
                fake_batch = generator(latent, index, alpha).detach()
                d_loss_fake = torch.mean(discriminator(fake_batch, index,
                                                       alpha))

                d_loss = d_loss_real + d_loss_fake
                d_optimizer.zero_grad()
                d_loss.backward() # if retain_graph=True
                # then gp works but I'm not sure it's right
                d_optimizer.step()

                # Compute gradient penalty
                alpha_gp = torch.rand(batch.size(0), 1, 1, 1).to(device)
                # mind that x_hat must be both detached from the previous
                # gradient graph (from fake_barch) and with
                # requires_graph=True so that the gradient can be computed
                x_hat = (alpha_gp * batch + (1 - alpha_gp) *
                         fake_batch).requires_grad_(True)
                # x_hat = torch.cuda.FloatTensor(x_hat).requires_grad_(True)
                out = discriminator(x_hat, index, alpha)
                grad = torch.autograd.grad(
                    outputs=out,
                    inputs=x_hat,
                    grad_outputs=torch.ones_like(out).to(device),
                    retain_graph=True,
                    create_graph=True,
                    only_inputs=True
                )[0]
                grad = grad.view(grad.size(0), -1) #is this the same as
                # detach?
                l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                d_loss_gp = torch.mean((l2norm - 1) ** 2)

                d_loss_gp *= lambda_gp
                d_optimizer.zero_grad()
                d_loss_gp.backward()
                d_optimizer.step()

                if (i + 1) % ncritic == 0:
                    latent = torch.randn(batch_size, num_channels, 1, 1).to(device)
                    fake_batch = generator(latent, index, alpha)
                    g_loss = - torch.mean(discriminator(fake_batch, index,
                                                      alpha))
                    g_optimizer.zero_grad()
                    g_loss.backward()
                    g_optimizer.step()

                # print debugging images
                if (i + 1) % debug_step == 0:
                    print_debugging_images(generator, debug_vectors,
                                           N, index, alpha, i)



def main():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True,
                        help="path to the dataset folder")
    parser.add_argument("--crop-size", type=int, default=128,
                        help="size of the cropped image around the face")
    parser.add_argument("--final-size", type=int, default=128,
                        help="final size of the desired images")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--alternating-step", type=int, default=10000,
                        help="step to alternate between fading and "
                             "stabilizing")
    parser.add_argument("--ncritic", type=int, default=1)
    parser.add_argument("--lambda-gp", type=float, default=0.1)
    parser.add_argument("--debug-step", type=int, default=100)
    dargs = parser.parse_args()

    if not os.path.exists("img/"):
        os.makedirs("img/")

    train(data_path=dargs.data_path,
          final_size=dargs.final_size,
          batch_size=dargs.batch_size,
          alternating_step=dargs.alternating_step,
          ncritic=dargs.ncritic,
          lambda_gp=dargs.lambda_gp,
          debug_step=dargs.debug_step)