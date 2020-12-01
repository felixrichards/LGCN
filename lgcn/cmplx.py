import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init as init
from lgcn.utils import _compress_shape, _recover_shape


def cmplx(real, imag):
    """Stacks real and imaginary component into single tensor.

    This functions more as a readability aid than a helper method.
    """
    return torch.stack([real, imag], dim=0)


def new_cmplx(real):
    """Creates a trivial complex tensor.
    """
    return cmplx(real, torch.zeros_like(real))


def magnitude(x, eps=1e-8, sq=False, **kwargs):
    """Computes magnitude of given complex tensor.

    Must return nonzero as grad(0)=inf
    """
    mag2 = x.pow(2).sum(dim=0)
    if sq:
        return mag2
    return torch.sqrt(torch.clamp(mag2, min=eps))


def phase(x, eps=1e-8, **kwargs):
    """Computes phase of given complex tensor.

    Must return nonzero as grad(0)=inf
    """
    return torch.atan(x[1] / torch.clamp(x[0], min=eps))


def concatenate(x, **kwargs):
    """Concatenates complex tensor into real tensor
    """
    return torch.cat([x[0], x[1]], dim=1)


def conv_cmplx(x, w, transpose=False, **kwargs):
    """Computes complex convolution.
    """
    conv = F.conv2d
    if transpose:
        conv = F.conv_transpose2d
        w = w.transpose(1, 2)

    w, wsh = _compress_shape(w)

    real = conv(x[0], w[0], **kwargs) - conv(x[1], w[1], **kwargs)
    imag = conv(x[0], w[1], **kwargs) + conv(x[1], w[0], **kwargs)

    return cmplx(real, imag)


def relu_cmplx(x, b=1e-8, inplace=False, **kwargs):
    """Computes complex relu.
    """
    r = magnitude(x, sq=False)
    if r.dim() < b.dim():
        b = b.flatten(0)
    elif r.dim() > b.dim():
        b = b.unsqueeze(-1)
    return F.relu(r + b) * x / r


def bnorm_cmplx_old(x, eps=1e-8):
    """Computes complex simple batch normalisation.
    """
    means = torch.mean(x, (1, 3, 4), keepdim=True)
    x = x - means

    stds = torch.std(magnitude(x, eps=eps, sq=False), (0, 2, 3), keepdim=True)
    x = x / torch.clamp(stds.unsqueeze(0), min=eps)

    return x


def pool_cmplx(x, kernel_size, **kwargs):
    """Computes complex pooling.
    """
    x, xs = _compress_shape(x)

    pool = F.avg_pool2d

    out = cmplx(
        pool(x[0], kernel_size, **kwargs),
        pool(x[1], kernel_size, **kwargs)
    )

    out = _recover_shape(out, xs)
    return out


def init_weights(w, mode='he'):
    """Initialises conv. weights according to C. Trabelsi, Deep Complex Networks
    """
    fan_in, fan_out = init._calculate_fan_in_and_fan_out(w[0])
    if mode == 'he':
        sigma = 1 / fan_in
    if mode == 'glorot':
        sigma = 1 / (fan_in + fan_out)

    mag = w[0].new_tensor(np.random.rayleigh(scale=sigma, size=w[0].size()))
    phase = w[0].new_tensor(np.random.uniform(low=-np.pi, high=np.pi, size=w[0].size()))

    with torch.no_grad():
        w.data = cmplx(mag * torch.cos(phase), mag * torch.sin(phase))
