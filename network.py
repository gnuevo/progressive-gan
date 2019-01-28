import torch
from torch import nn
from collections import OrderedDict
import math

"""
ModuleList(
 block0
 block1
 ...
 blockN
)
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Flatten(nn.Module):
    """
    https://discuss.pytorch.org/t/flatten-layer-of-pytorch-build-by-sequential-container/5983#post_2
    """
    def forward(self, input):
        return input.view(input.size(0), -1)

class Generator(nn.Module):
    # set the maximum number of channels to 512
    max_channels = 512

    def __init__(self, final_size=1024):
        """

        Args:
            num_channels: latent space size
        """
        super(Generator, self).__init__()
        self.final_size = final_size
        self.num_blocks = int(math.log(final_size, 2) - 1)
        self.num_channels = 2**(self.num_blocks - 1 + 4)
        self.network = nn.ModuleList()
        self.toRGB = nn.ModuleList() # stores the toRGB modules for
        # different resolutions
        self.upsample = nn.Upsample(scale_factor=2)

    def _get_block_by_index(self, index):
        """Returns a block of the network given its index
        Blocks are constructed according to the original paper
        https://research.nvidia.com/publication/2017-10_Progressive-Growing-of

        Args:
            index: index of the block 0..(n-1)
            num_channels: size of the latent space

        Returns: list of layers with names to be put in an OrderedDict
        """

        if index == 0:
            # the first block receives the latent vector
            channels = min(int(2 ** (self.num_blocks - index + 4 - 1)),
                           self.max_channels)
            block = [
                # ('linear_0', nn.Linear(channels, 4*4*channels)),
                ('conv_0_0', nn.Conv2d(channels, channels, 4, padding=3)),
                ('lrelu_0_0', nn.LeakyReLU(negative_slope=0.2)),
                ('conv_0_1', nn.Conv2d(channels, channels, 3, padding=1)),
                ('lrelu_0_1', nn.LeakyReLU(negative_slope=0.2)),
            ]
        else:
            # each block has one Upsample and 2 Conv2d
            in_channels = min(int(2**(self.num_blocks - index + 5 - 1)),
                           self.max_channels)
            out_channels = min(int(2 ** (self.num_blocks - index + 4 - 1)),
                              self.max_channels)
            block = [
                ('upsample_{}'.format(index), nn.Upsample(scale_factor=2)),
                ('conv_{}_0'.format(index),
                    nn.Conv2d(in_channels, out_channels, 3, padding=1)),
                ('lrelu_{}_0'.format(index), nn.LeakyReLU(negative_slope=0.2)),
                ('conv_{}_1'.format(index),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1)),
                ('lrelu_{}_1'.format(index), nn.LeakyReLU(negative_slope=0.2))
            ]
            # if index == self.num_blocks:
            #     # the last block has a last 1x1 convolution
            #     block.append(
            #         ('conv_{}_2'.format(index), nn.Conv2d(out_channels, 3, 3))
            #     )
        return nn.Sequential(OrderedDict(block))

    def _get_toRGB_by_index(self, index):
        """Returns the toRGB module by index

        Images at different resolutions must be converted to RGB before send
        send them to the discriminator. The toRGB modules are comprised of
        1x1 convolutions convert an image of arbitrary size and N channels
        to an image of the same size and 3 channels (RGB).

        Args:
            index: index value

        Returns:
        """
        channels = min(int(2 ** (self.num_blocks - index - 1 + 4)),
                       self.max_channels)
        toRGB = nn.Conv2d(channels, 3, 1)
        return toRGB

    def generate_network(self):
        network = nn.ModuleList()
        toRGB = nn.ModuleList()
        for n in range(self.num_blocks):
            network.append(self._get_block_by_index(n))
            toRGB.append(self._get_toRGB_by_index(n))
        # print('----------------------------------------------------')
        # print('----------------------------------------------------')
        # print('----------------------------------------------------')
        print("Generator")
        print(network)
        print(toRGB)
        self.network = network.to(device)
        self.toRGB = toRGB.to(device)

    def forward(self, latent, index, alpha):
        """

        Args:
            latent: input latent vector
            index: 0..n+1, indicates which parts of the network must be run
            during training.
                0: runs the first (4x4) block
                1: runs the first and second (8x8) blocks, including alpha
                ...
            alpha: alpha value as in the paper

        Returns:

        """
        x = latent
        for i in range(index+1):
            prev_x = x
            x = self.network[i](x)
        if index > 0 and i == index:
            upsampled = self.upsample(prev_x)
            img = (1 - alpha) * self.toRGB[index-1](upsampled) \
                  + alpha * self.toRGB[index](x)
        else:
            img = self.toRGB[index](x)
        return img


class Discriminator(nn.ModuleList):
    # set the maximum number of channels to 512
    max_channels = 512

    def __init__(self, final_size=1024):
        super(Discriminator, self).__init__()
        self.final_size = final_size
        self.num_blocks = int(math.log(final_size, 2) - 1)
        self.num_channels = 2 ** (self.num_blocks - 1 + 4)
        self.network = nn.ModuleList()
        self.fromRGB = nn.ModuleList()  # stores the toRGB modules for
        # different resolutions
        self.avgpool = nn.AvgPool2d(kernel_size=2)

    def _get_block_by_index(self, index):
        if index == 0:
            channels = min(int(2 ** (self.num_blocks - index + 4 - 1)),
                           self.max_channels)
            # the first convolution receives `channels + 1` to account for
            # the minibatch std
            block = [
                ('conv_0_0', nn.Conv2d(channels + 1, channels, 3, padding=1)),
                ('lrelu_0_0', nn.LeakyReLU(negative_slope=0.2)),
                ('conv_0_1', nn.Conv2d(channels, channels, 4, padding=0)),
                ('lrelu_0_1', nn.LeakyReLU(negative_slope=0.2)),
                ('flatten_0', Flatten()),
                ('linear_0', nn.Linear(channels, 1))
            ]
        else:
            # each block has one Upsample and 2 Conv2d
            in_channels = min(int(2 ** (self.num_blocks - index + 4 - 1)),
                              self.max_channels)
            out_channels = min(int(2 ** (self.num_blocks - index + 5 - 1)),
                              self.max_channels)
            block = [
                ('conv_{}_0'.format(index),
                    nn.Conv2d(in_channels, in_channels, 3, padding=1)),
                ('lrelu_{}_0'.format(index), nn.LeakyReLU(negative_slope=0.2)),
                ('conv_{}_1'.format(index),
                    nn.Conv2d(in_channels, out_channels, 3, padding=1)),
                ('lrelu_{}_1'.format(index), nn.LeakyReLU(negative_slope=0.2)),
                ('avgpool_{}'.format(index), nn.AvgPool2d(2))
            ]
        return nn.Sequential(OrderedDict(block))

    def _get_fromRGB_by_index(self, index):
        channels = min(int(2 ** (self.num_blocks - index - 1 + 4)),
                       self.max_channels)
        fromRGB = nn.Conv2d(3, channels, 1)
        return fromRGB

    def generate_network(self):
        network = nn.ModuleList()
        fromRGB = nn.ModuleList()
        for n in range(self.num_blocks):
            network.append(self._get_block_by_index(n))
            fromRGB.append(self._get_fromRGB_by_index(n))
        print("Discriminator")
        print(network)
        print(fromRGB)
        self.network = network.to(device)
        self.fromRGB = fromRGB.to(device)

    def forward(self, img, index, alpha):
        """

        Args:
            img:
            index:
            alpha:

        Returns:

        """
        averaged = self.fromRGB[index-1](self.avgpool(img))
        x = self.fromRGB[index](img)
        for i in reversed(range(index+1)):
            if i == 0:
                # compute the minibatch std
                minibatch_std = img.std(dim=0)
                averaged_std = minibatch_std.mean()
                # concat minibatch std to tensor
                broadcasted_std = torch.ones(img.shape[0], 1, 4,
                                             4).to(device) * averaged_std
                x = torch.cat((x, broadcasted_std), 1)
            x = self.network[i](x)
            if index > 0 and i == index:
                x = (1 - alpha) * averaged + alpha * x
        return x
