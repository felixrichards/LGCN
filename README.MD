# Learnable Gabor modulated complex-valued networks for orientation robustness

This source code provides the implementation for [Learnable Gabor-modulated Complex-valued Networks (LGCNs)](https://arxiv.org/abs/2011.11734).

Authors: Felix Richards, Adeline Paiement, Xianghua Xie, Elisabeth Sola, Pierre-Alain Duc.

### Usage

The model used in the paper is found in `./lgcn/models.py`. PyTorch modules implementing real and complex layers are found in `./lgcn/modules.py` and `./lgcn/cmplx_modules.py`, respectively.

### Dependencies

The following packages are used in the main source folder `./lgcn/`:

```
torch
numpy
```

There is a demo file `./demo.py` that loads and trains the model used in the paper on rotated MNIST, though this uses my own prototyping package [quicktorch](https://github.com/felixrichards/quicktorch).

### Issues

There may (read definitely) be bugs with the real-only implementation.

### Contact

Felix A. Richards <br/>
Swansea University <br/>
felixarichards@gmail.com <br/>
