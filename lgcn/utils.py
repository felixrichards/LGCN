import torch


def _pair(x):
    if type(x) is int:
        return (x, x)
    return x


def _compress_shape(x):
    xs = None
    if x.dim() == 6:
        xs = x.size()
        x = x.view(
            2,
            xs[1] * xs[2],
            xs[3],
            *xs[4:]
        )

    return x, xs


def _recover_shape(x, xs):
    if xs is not None:
        x = x.view(
            *xs[:4],
            x.size(-2),
            x.size(-1)
        )

    return x
