import torch
import scipy
from datasets import NoduleDataset 
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import random
import json
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn_v2 as fcnn
import torch
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision.models.detection._utils as det_utils


device = torch.device("cuda")
cpu_device = torch.device("cpu")

model = torch.load(PATH).to(device)

transform = transforms.Compose([
    transforms.Resize(1024),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

val_dataset = NoduleDataset("./refineddataset/testxrays", "./refineddataset/control", "./refineddataset/nodules.json", 0, transform)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)


for images, targets in val_loader:
        images = list(image.to(device) for image in images)
        
        with torch.no_grad():
            outputs = model(images)
        
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        