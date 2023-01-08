#!/usr/bin/env python
import os
import torch
from torch import nn


PACKAGE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.split(PACKAGE)[0], 'data')


model = nn.Linear(1, 1).eval()
model = torch.jit.trace(model, example_inputs=torch.rand(1, 1))
output_path = os.path.join(DATA, 'trace_linear.pt')
model.save(output_path)
print(f"saved model to {output_path}")
