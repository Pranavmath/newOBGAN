from datasets import NoduleDataset, CurriculumNoduleDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import random
import json
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn_v2 as fcnn
import torch
import wandb
from torchmetrics.detection import MeanAveragePrecision, IntersectionOverUnion
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision.models.detection._utils as det_utils
import numpy as np 
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt 
from matplotlib import cm


transform = transforms.Compose([
    transforms.Resize(1024),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

device = torch.device("cuda")
cpu_device = torch.device("cpu")

model = torch.load(PATH).to(device)

annotations = json.load(open("./refineddataset/nodules.json"))
metadata = json.load(open("./refineddataset/nodulemetada.json"))

metric = IntersectionOverUnion()

areas, brightnesses, ious = [], [], []

for image, nodules in annotations.items():
    image = Image.open(image + ".jpg")
    image = transform(image)

    with torch.no_grad():
        outputs = model(image)

    outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

    for nodule_name, nodule_bbox in nodules:
        _, area, brightness = metadata[nodule_name]

        nodule_target = {
            "boxes": torch.tensor(nodule_bbox, dtype=torch.float32),
            "labels": torch.ones(1, dtype=torch.int64),  # Assuming label for all boxes is 1 (nodule)
        }

        iou = metrics(preds=outputs, target=targets)["iou"]

        areas.append(area)
        brightnesses.append(brightness)
        ious.append(iou)


data = zip(areas, brightnesses)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(areas, brightnesses, iou, cmap=cm.Blues)
plt.show()