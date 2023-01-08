#!/usr/bin/env python
import os
import torch
from torch import nn


PACKAGE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.split(PACKAGE)[0], 'data')


model = nn.Identity().eval()
model = torch.jit.script(model)
output_path = os.path.join(DATA, 'script_identity.pt')
model.save(output_path)
print(f"saved model to {output_path}")
