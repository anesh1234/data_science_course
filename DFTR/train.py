import torch

model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
print(model)
