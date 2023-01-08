#!/usr/bin/env python
import os
import torch
import torchvision as tv


PACKAGE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.split(PACKAGE)[0], 'data')

# http://pytorch.org/vision/master/models/faster_rcnn.html

# Faster R-CNN model with a ResNet-50-FPN backbone from the Faster R-CNN:
# Towards Real-Time Object Detection with Region Proposal Networks paper.
model = tv.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval()
model = torch.jit.script(model)
output_path = os.path.join(DATA, 'fasterrcnn_resnet50_fpn.pt')
model.save(output_path)
print(f"saved model to {output_path}")

# # Constructs an improved Faster R-CNN model with a ResNet-50-FPN backbone from
# # Benchmarking Detection Transfer Learning with Vision Transformers paper.
# model = tv.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=True).eval()
# model = torch.jit.script(model)
# output_path = os.path.join(DATA, 'fasterrcnn_resnet50_fpn_v2.pt')
# model.save(output_path)
# print(f"saved model to {output_path}")

# Constructs a high resolution Faster R-CNN model with a
# MobileNetV3-Large FPN backbone.
model = tv.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True).eval()
model = torch.jit.script(model)
output_path = os.path.join(DATA, 'fasterrcnn_mobilenet_v3_large_fpn.pt')
model.save(output_path)
print(f"saved model to {output_path}")

# Low resolution Faster R-CNN model with a MobileNetV3-Large backbone tunned
# for mobile use cases.
model = tv.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True).eval()
model = torch.jit.script(model)
output_path = os.path.join(DATA, 'fasterrcnn_mobilenet_v3_large_320_fpn.pt')
model.save(output_path)
print(f"saved model to {output_path}")
