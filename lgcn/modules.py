import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.conv import Conv2d
from .gabor import GaborFunction
from .utils import _pair


class Gabor(nn.Module):
    """Wraps the Gabor implementation into a NN layer w/o convolution.

    Args:
        no_g (int, optional): The number of desired Gabor filters.
        layer (boolean, optional): Whether this is used as a layer or a
            modulation function.
    """
    def __init__(self, no_g=4, layer=False, kernel_size=None, **kwargs):
        super().__init__(**kwargs)
        self.theta = nn.Parameter(data=torch.Tensor(no_g))
        self.theta.data = torch.arange(no_g) / (no_g) * np.pi
        self.register_parameter(name="theta", param=self.theta)
        self.l = nn.Parameter(data=torch.Tensor(no_g))
        self.l.data.uniform_(-1 / torch.sqrt(no_g),
                             1 / torch.sqrt(no_g))
        self.register_parameter(name="lambda", param=self.l)
        self.register_buffer("gabor_filters", torch.Tensor(no_g, 1, 1,
                                                           *kernel_size))
        self.GaborFunction = GaborFunction.apply

        self.no_g = no_g

    def forward(self, x):
        print(f'x.size()={x.size()}')

        out = self.GaborFunction(
            x,
            torch.stack((
                self.theta,
                self.l
            ))
        )

        print(f'out.size()={out.size()}')

        out = out.view(-1 , *out.size()[2:])

        print(f'out.size()={out.size()}')

        return out


class LGConv(Conv2d):
    """Implements a convolutional layer where weights are first Gabor modulated.

    Args:
        input_features (torch.Tensor): Feature channels in.
        output_features (torch.Tensor): Feature channels out.
        kernel_size (int, tuple): Size of kernel.
        no_g (int, optional): The number of desired Gabor filters.
        gabor_pooling (bool, optional): Whether to apply max pooling along the
            gabor axis. Defaults to False.
        conv_kwargs (dict, optional): Contains keyword arguments to be passed
            to convolution operator. E.g. stride, dilation, padding.
    """
    def __init__(self, input_features, output_features, kernel_size,
                 gabor_pooling=None, no_g=2, **conv_kwargs):
        if not max_gabor:
            if output_features % no_g:
                raise ValueError("Number of gabor filters ({}) does not divide output features ({})"
                                 .format(str(no_g), str(output_features)))
            output_features //= no_g
        kernel_size = _pair(kernel_size)
        super().__init__(input_features, output_features, kernel_size, **conv_kwargs)

        self.conv = F.conv2d
        self.conv_kwargs = conv_kwargs
        self.gabor = Gabor(no_g, kernel_size=kernel_size)
        self.no_g = no_g
        if gabor_pooling:
            self.gabor_pooling = torch.max

    def forward(self, x):
        enhanced_weight = self.gabor(self.weight)
        out = self.conv(x, enhanced_weight, **self.conv_kwargs)

        if self.gabor_pooling is None:
            return out

        pool_out = out.view(
            out.size(1),
            enhanced_weight.size(1) // self.no_g,
            self.no_g,
            out.size(3),
            out.size(4)
        )

        pool_out, max_idxs = self.gabor_pooling(pool_out, dim=2)
        out = pool_out

        return out
