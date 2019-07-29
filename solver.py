import torch
from torch.optim import Adam
from torchvision.utils import save_image
from torch.nn import Upsample
from network import Generator, Discriminator
from data_loader import get_loader
import time
import datetime
import os
import json
from collections import OrderedDict
from logger import Logger

class Solver(object):

    def __init__(self, configuration):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # retrieve configuration variables
        self.data_path = configuration.data_path
        self.crop_size = configuration.crop_size
        self.final_size = configuration.final_size
        self.batch_size = configuration.batch_size
        self.alternating_step = configuration.alternating_step
        self.ncritic = configuration.ncritic
        self.lambda_gp = configuration.lambda_gp
        self.debug_step = configuration.debug_step
        self.save_step = configuration.save_step
        self.max_checkpoints = configuration.max_checkpoints
        self.log_step = configuration.log_step
        self.tflogger = Logger(configuration.log_dir)
        ## directoriess
        self.train_dir = configuration.train_dir
        self.img_dir = configuration.img_dir
        self.models_dir = configuration.models_dir
        ## variables
        self.eps_drift = 0.001

        self._initialise_networks()

    def _initialise_networks(self):
        self.generator = Generator(final_size=self.final_size)
        self.generator.generate_network()
        self.g_optimizer = Adam(self.generator.parameters())

        self.discriminator = Discriminator(final_size=self.final_size)
        self.discriminator.generate_network()
        self.d_optimizer = Adam(self.discriminator.parameters())

        self.num_channels = min(self.generator.num_channels,
                                self.generator.max_channels)
        self.upsample = [Upsample(scale_factor=2**i)
                for i in reversed(range(self.generator.num_blocks))]

    def print_debugging_images(self, generator, latent_vectors, shape, index,
                               alpha, iteration):
        with torch.no_grad():
            columns = []
            for i in range(shape[0]):
                row = []
                for j in range(shape[1]):
                    img_ij = generator(latent_vectors[i * shape[1] +
                                                      j].unsqueeze_(0),
                                       index, alpha)
                    img_ij = self.upsample[index](img_ij)
                    row.append(img_ij)
                columns.append(torch.cat(row, dim=3))
            debugging_image = torch.cat(columns, dim=2)
        # denorm
        debugging_image = (debugging_image + 1) / 2
        debugging_image.clamp_(0, 1)
        save_image(debugging_image.data,
                   os.path.join(self.img_dir, "debug_{}_{}.png".format(index,
                                                                      iteration)))

    def save_trained_networks(self, block_index, phase, step):
        models_file = os.path.join(self.models_dir, "models.json")
        if os.path.isfile(models_file):
            with open(models_file, 'r') as file:
                models_config = json.load(file)
        else:
            models_config = json.loads('{ "checkpoints": [] }')

        generator_save_name = "generator_{}_{}_{}.pth".format(
                                    block_index, phase, step
                                )
        torch.save(self.generator.state_dict(),
                   os.path.join(self.models_dir, generator_save_name))

        discriminator_save_name = "discriminator_{}_{}_{}.pth".format(
                                    block_index, phase, step
                                )
        torch.save(self.discriminator.state_dict(),
                   os.path.join(self.models_dir, discriminator_save_name))

        models_config["checkpoints"].append(OrderedDict({
            "block_index": block_index,
            "phase": phase,
            "step": step,
            "generator": generator_save_name,
            "discriminator": discriminator_save_name
        }))
        if len(models_config["checkpoints"]) > self.max_checkpoints:
            old_save = models_config["checkpoints"][0]
            os.remove(os.path.join(self.models_dir, old_save["generator"]))
            os.remove(os.path.join(self.models_dir, old_save["discriminator"]))
            models_config["checkpoints"] = models_config["checkpoints"][1:]
        with open(os.path.join(self.models_dir, "models.json"), 'w') as file:
            json.dump(models_config, file, indent=4)

    def train(self):
        # get debugging vectors
        N = (5, 10)
        debug_vectors = torch.randn(N[0] * N[1], self.num_channels, 1,
                                    1).to(self.device)

        # get loader
        loader = get_loader(self.data_path, self.crop_size, self.batch_size)

        losses = {
            "d_loss_real": None,
            "d_loss_fake": None,
            "g_loss": None
        }

        # training loop
        start_time = time.time()
        absolute_step = -1
        for index in range(self.generator.num_blocks):
            loader.dataset.set_transform_by_index(index)
            data_iterator = iter(loader)
            for phase in ('fade', 'stabilize'):
                if index == 0 and phase == 'fade': continue
                if phase == 'phade': self.alternating_step = 10000 #FIXME del
                print("index: {}, size: {}x{}, phase: {}".format(
                    index, 2 ** (index + 2), 2 ** (index + 2), phase))
                for i in range(self.alternating_step):
                    absolute_step += 1
                    try:
                        batch = next(data_iterator)
                    except:
                        data_iterator = iter(loader)
                        batch = next(data_iterator)

                    alpha = i / self.alternating_step if phase == "fade" else 1.0

                    batch = batch.to(self.device)

                    d_loss_real = - torch.mean(
                        self.discriminator(batch, index, alpha))
                    losses["d_loss_real"] = torch.mean(d_loss_real).data[0]

                    latent = torch.randn(
                        batch.size(0), self.num_channels, 1, 1).to(self.device)
                    fake_batch = self.generator(latent, index, alpha).detach()
                    d_loss_fake = torch.mean(
                        self.discriminator(fake_batch, index, alpha))
                    losses["d_loss_fake"] = torch.mean(d_loss_fake).data[0]

                    # drift factor
                    drift = d_loss_real.pow(2) + d_loss_fake.pow(2)

                    d_loss = d_loss_real + d_loss_fake + self.eps_drift * drift
                    self.d_optimizer.zero_grad()
                    d_loss.backward()  # if retain_graph=True
                    # then gp works but I'm not sure it's right
                    self.d_optimizer.step()

                    # Compute gradient penalty
                    alpha_gp = torch.rand(batch.size(0), 1, 1, 1).to(self.device)
                    # mind that x_hat must be both detached from the previous
                    # gradient graph (from fake_barch) and with
                    # requires_graph=True so that the gradient can be computed
                    x_hat = (alpha_gp * batch + (1 - alpha_gp) *
                             fake_batch).requires_grad_(True)
                    # x_hat = torch.cuda.FloatTensor(x_hat).requires_grad_(True)
                    out = self.discriminator(x_hat, index, alpha)
                    grad = torch.autograd.grad(
                        outputs=out,
                        inputs=x_hat,
                        grad_outputs=torch.ones_like(out).to(self.device),
                        retain_graph=True,
                        create_graph=True,
                        only_inputs=True
                    )[0]
                    grad = grad.view(grad.size(0), -1)  # is this the same as
                    # detach?
                    l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                    d_loss_gp = torch.mean((l2norm - 1) ** 2)

                    d_loss_gp *= self.lambda_gp
                    self.d_optimizer.zero_grad()
                    d_loss_gp.backward()
                    self.d_optimizer.step()

                    # train generator
                    if (i + 1) % self.ncritic == 0:
                        latent = torch.randn(
                            self.batch_size, self.num_channels, 1, 1).to(self.device)
                        fake_batch = self.generator(latent, index, alpha)
                        g_loss = - torch.mean(self.discriminator(
                                                    fake_batch, index, alpha))
                        losses["g_loss"] = torch.mean(g_loss).data[0]
                        self.g_optimizer.zero_grad()
                        g_loss.backward()
                        self.g_optimizer.step()

                    # tensorboard logging
                    if (i + 1) % self.log_step == 0:
                        elapsed = time.time() - start_time
                        elapsed = str(datetime.timedelta(seconds=elapsed))
                        print("{}:{}:{}/{} time {}, d_loss_real {}, "
                              "d_loss_fake {}, "
                              "g_loss {}, alpha {}".format(index, phase, i,
                                                           self.alternating_step,
                                                           elapsed,
                                                           d_loss_real,
                                              d_loss_fake,
                                              g_loss, alpha))
                        for name, value in losses.items():
                            self.tflogger.scalar_summary(name, value, absolute_step)


                    # print debugging images
                    if (i + 1) % self.debug_step == 0:
                        self.print_debugging_images(
                            self.generator, debug_vectors, N, index, alpha, i)

                    # save trained networks
                    if (i + 1) % self.save_step == 0:
                        self.save_trained_networks(index, phase, i)