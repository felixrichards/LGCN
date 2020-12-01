import math
import torch
from torch.autograd import Function
from .cmplx import cmplx


class GaborFunction(Function):
    """Extends autograd Function to create a Gabor filter with learnable theta.
    """

    @staticmethod
    def forward(ctx, input, weight):
        """Applies a Gabor filter to given input. Weight contains thetas/sigmas.

        Args:
            input (Tensor): data to apply filter to.
            weight (Tensor): theta and sigma parameters.
                Must have weight.size() = [N, 2]
        """
        output = gabor(input, weight).unsqueeze(1).unsqueeze(1)
        ctx.save_for_backward(input, weight, output)
        return match_shape(output * input, input)

    @staticmethod
    def backward(ctx, grad_output):
        """Computes gradients for Gabor filter backprop.

        Args:
            grad_output (Tensor): gradient from graph.
        """
        input, weight, result = ctx.saved_tensors
        grad_weight = gabor_gradient(input, weight).unsqueeze(2).unsqueeze(2)
        grad_output = match_shape(grad_output, grad_weight, False)
        return (
            result * grad_output,
            (input * grad_weight * grad_output).permute(5, 4, 3, 2, 0, 1),
        )


class GaborFunctionCmplx(Function):
    """Extends autograd Function to create a Gabor filter with learnable theta.
    """

    @staticmethod
    def forward(ctx, weight, gabor_params):
        """Applies a Gabor filter to given weight. gabor_params contains thetas/sigmas.

        Args:
            weight (Tensor): data to apply filter to.
            gabor_params (Tensor): theta and sigma parameters.
                Must have gabor_params.size() = [N, 2]
        """
        gabor_filter = gabor_cmplx(weight, gabor_params)
        ctx.save_for_backward(weight, gabor_params, gabor_filter)
        # print(f'weight.size()={weight.size()}, gabor_filter.size()={gabor_filter.size()}')
        return weight * gabor_filter

    @staticmethod
    def backward(ctx, grad_output):
        """Computes gradients for Gabor filter backprop.

        Args:
            grad_output (Tensor): gradient from graph.
        """
        weight, gabor_params, gabor_filter = ctx.saved_tensors
        grad_gabor = gabor_gradient_cmplx(weight, gabor_params).unsqueeze(3).unsqueeze(2)
        # print(f'grad_gabor.size()={grad_gabor.size()}, weight.size()={weight.size()}, grad_output.size()={grad_output.size()}, gabor_filter.size()={gabor_filter.size()}')
        grad = (grad_gabor * weight * grad_output)
        # print(f'grad.size()={grad.size()}')
        return (
            gabor_filter * grad_output,
            grad.permute(0, 2, 4, 5, 6, 1, 3),
        )


def match_shape(x, y, compress=True):
    """Reshapes a tensor to be broadcastable with another

    The input tensor, x, by default will be reshaped so that all but the first
    dimensions match all but the first dimensions of y.
    Args:
        compress (boolean): If false x will be reshaped so that it's first
            dimension is split into two, with the first matching that of y.

    Returns:
        A reshaped tensor
    """
    if compress:
        x = x.view(-1, *y.size()[1:])
    else:
        x = x.view(y.size(1), -1, *x.size()[1:])
    return x


def cartesian_coords(weight):
    """Generates cartesian coordinates for analytical filter.

    Args:
        weight: The weight to be passed through the filter. This is used to
            set compatible tensor properties, e.g. height, width, device.

    Returns:
        torch.tensor: x coordinates
        torch.tensor: y coordinates
    """
    h = weight.size(-2)
    w = weight.size(-1)
    y, x = torch.meshgrid([torch.arange(-h / 2, h / 2), torch.arange(-w / 2, w / 2)])
    x = x.to(weight.device)
    y = y.to(weight.device)
    return x, y


def norm(t, eps=1e-12):
    """Normalises tensor between 0 and 1
    """
    return (t - t.min()) / (t.max() - t.min() + eps)


def gabor(weight, params):
    """Computes a gabor filter.

    Args:
        weight: The weight to be passed through the filter. This is used to
            set compatible tensor properties, e.g. height, width, device.
        params: theta and sigma parameters.
            Must have params.size() = [G, 2].
            Here G = no of gabor filters.
            params[:, 0] = theta parameters.
            params[:, 1] = sigma parameters.

    Returns:
        torch.tensor: gabor filter with (F_out*G, F_in, H, W) dimensions
    """
    x, y = cartesian_coords(weight)
    theta = params[0]
    l = params[1].unsqueeze(1).unsqueeze(1)
    x_p = x_prime(x, y, theta)
    return norm(f_h(x, y) * s_h(x_p, l))


def gabor_cmplx(weight, params):
    """Computes a complex gabor filter.

    Args:
        weight: The weight to be passed through the filter. This is used to
            set compatible tensor properties, e.g. height, width, device.
        params: theta and sigma parameters.
            Must have params.size() = [G, 2].
            Here G = no of gabor filters.
            params[:, 0] = theta parameters.
            params[:, 1] = sigma parameters.

    Returns:
        torch.tensor: gabor filter with (2, 1, G, 1, H, W) dimensions
    """
    x, y = cartesian_coords(weight)
    f = f_h(x, y)
    theta = params[0]
    l = params[1].unsqueeze(1).unsqueeze(1)
    x_p = x_prime(x, y, theta)

    real = f * s_h(x_p, l)
    imag = f * s_h_imag(x_p, l)

    return norm(cmplx(real, imag)).unsqueeze(2).unsqueeze(1)


def gabor_gradient(weight, params):
    """Computes a gabor filter derivative at a given weight.

    Args:
        weight: The weight to be passed through the filter
        params: theta and sigma parameters.
            Must have params.size() = [N, 2].
            Here N = no of gabor filters.
            params[:, 0] = theta parameters.
            params[:, 1] = sigma parameters.

    Returns:
        torch.tensor: gabor gradient with (F_out*G, F_in, H, W) dimensions
    """
    x, y = cartesian_coords(weight)
    theta = params[0]
    l = params[1].unsqueeze(1).unsqueeze(1)
    return f_h(x, y) * d_s_h(x, y, theta, l)


def gabor_gradient_cmplx(weight, params):
    """Computes a complex gabor filter derivative at a given weight.

    Args:
        weight: The weight to be passed through the filter
        params: theta and sigma parameters.
            Must have params.size() = [N, 2].
            Here N = no of gabor filters.
            params[:, 0] = theta parameters.
            params[:, 1] = sigma parameters.

    Returns:
        torch.tensor: complex gabor gradient with (2, N, 2, H, W) dimensions
    """
    x, y = cartesian_coords(weight)
    theta = params[0]
    l = params[1].unsqueeze(1).unsqueeze(1)
    f = f_h(x, y)
    x_p = x_prime(x, y, theta)
    y_p = y_prime(x, y, theta)
    dx = -2 * math.pi / l
    df = dx * f
    dgdt = cmplx(
        - df * y_p * torch.sin(dx * x_p),
        df * y_p * torch.cos(dx * x_p),
    )
    dgdl = cmplx(
        df / l * x_p * torch.sin(dx * x_p),
        - df / l * x_p * torch.cos(dx * x_p),
    )
    return torch.stack([dgdt, dgdl], dim=1)



def f_h(x, y):
    """First half of filter
    """
    return torch.exp(-(x ** 2 + y ** 2)).unsqueeze(0)


def s_h(x_p, l):
    """Second half of filter
    """
    return torch.cos(2 * math.pi / l * x_p)


def s_h_imag(x_p, l):
    """Second half of filter's imaginary component
    """
    return torch.sin(2 * math.pi / l * x_p)


def d_s_h(x, y, theta, l):
    """First half of filter derivative
    """
    x_p = x_prime(x, y, theta)
    dx = torch.sin(2 * math.pi / l * x_p)
    dt = -2 * math.pi / l * y_prime(x, y, theta) * dx
    dl = 2 * math.pi / l ** 2 * x_p * dx
    return torch.stack([dt, dl])


def x_prime(x, y, theta):
    """Computes x*cos(theta) + y*sin(theta)
    """
    return (
        torch.cos(theta).unsqueeze(1).unsqueeze(1) * x
        + torch.sin(theta).unsqueeze(1).unsqueeze(1) * y
    )


def y_prime(x, y, theta):
    """Computes y*cos(theta) - x*sin(theta)
    """
    return (
        torch.cos(theta).unsqueeze(1).unsqueeze(1) * y
        - torch.sin(theta).unsqueeze(1).unsqueeze(1) * x
    )
