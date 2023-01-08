#!/usr/bin/env python
import os
import torch
import torchvision as tv


PACKAGE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.split(PACKAGE)[0], 'data')


model = tv.models.resnet18(pretrained=True).eval()
model = torch.jit.trace(model, example_inputs=torch.rand(1, 3, 224, 224))
output_path = os.path.join(DATA, 'trace_resnet18.pt')
model.save(output_path)
print(f"saved model to {output_path}")
