import torch
import torch.nn as nn
import torch.optim as optim
from quicktorch.utils import train, evaluate
from quicktorch.data import mnist, cifar
from lgcn.models import LGCN


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
idxs = (torch.arange(50000), torch.arange(50000, 60000))
train_loader, val_loader, test_loader = mnist(batch_size=512, rotate=True, idxs=idxs)

model = LGCN(1, 10, no_g=4).to(device)

optimizer = optim.Adam(model.parameters())
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
train(model, [train_loader, val_loader], epochs=1, opt=optimizer,
      device=device, sch=scheduler)

evaluate(model, test_loader, device=device)
