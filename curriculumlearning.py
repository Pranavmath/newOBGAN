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
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision.models.detection._utils as det_utils
import numpy as np

wandb.init(project="diff model training", save_code=True)
wandb.save("./curriculumlearning.py")

def is_negative_target(targets):
    # needs to be a evaluation  batch size of 1 so 1 target only
    if(len(targets) != 1):
      raise ValueError("The length of targets should be 1")
    target = targets[0]

    assert len(target["labels"]) == len(target["boxes"])
    return 0 == len(target["labels"]) == len(target["boxes"])

def collate_fn(batch):
    return tuple(zip(*batch))

def get_model():
    model = fcnn(weights="DEFAULT", rpn_batch_size_per_image=256, rpn_positive_fraction=0.2)
    num_classes = 2  # 1 class (nodule) + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    anchor_sizes = ((8,), (16,), (32,), (64,), (75,))  # Reduce default sizes
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    model.rpn.anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios
    )
    
    model.roi_heads.box_predictor.loss_cls = torch.nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 5.0])  # Adjust class weights
    )

    return model


transform = transforms.Compose([
    transforms.Resize(1024),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

# num of epochs per difficulty
NUM_EPOCHS = 1
# doing batch size of 4 since 1, 2, or 4 was recommended for faster rcnn
BATCH_SIZE = 4
SAVE_MODEL_INTERVAL = 12

# we train on the NUM_HARD hardest images for WARMUP number of epochs
NUM_HARD = 30
WARMUP = 20
# NUM_ADD is the number of images we add every diff step
NUM_ADD = 10

device = torch.device("cuda")
cpu_device = torch.device("cpu")

model = get_model().to(device)

train_dataset = CurriculumNoduleDataset("./refineddataset/trainxrays", "./refineddataset/control", "./refineddataset/nodules.json", "./refineddataset/difficulties.json", 0, transform)
train_dataset.difficulty_step(NUM_HARD)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

val_dataset = NoduleDataset("./refineddataset/testxrays", "./refineddataset/control", "./refineddataset/nodules.json", 0, transform)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)


params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(
    params,
    lr=0.005,  # Start with a lower LR
    momentum=0.9,
    weight_decay=1e-4
)

scaler = torch.amp.GradScaler()

wandb.watch(model)

metric = MeanAveragePrecision(iou_type="bbox")

print("Started Training")

do_steps = [False] * WARMUP + [True] * 1000

for step_idx, do_step in enumerate(do_steps):
    print(f"length in batchs: {len(train_loader)}")
    model.train()

    for epoch in range(NUM_EPOCHS):
        avg_train_loss = []

        for images, targets in train_loader:
            optimizer.zero_grad()

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            scaler.scale(losses).backward()
            
            #"""
            scaler.unscale_(optimizer)

            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            #"""

            scaler.step(optimizer)
            scaler.update()

            avg_train_loss.append(losses.item())



    print(f"Evaluating for length (in batchs): {len(train_loader)}")

    model.eval()

    total_negative, correct_negative = 0, 0

    for images, targets in val_loader:
        images = list(image.to(device) for image in images)
        
        with torch.no_grad():
            outputs = model(images)
        
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        
        if not is_negative_target(targets):
            metric.update(outputs, targets)
        else:
            total_negative += 1
            if is_negative_target(outputs): correct_negative += 1
    
    if total_negative == 0:
        control_accuracy = -1
    else:
        control_accuracy = correct_negative/total_negative
    
    results = metric.compute()
    metric.reset()

    # first - iou=0.5:0.95, second - iou=0.50, third - iou=0.75
    iou, iou50, iou75 = results["map"], results["map_50"], results["map_75"]
    avg_train_loss = sum(avg_train_loss)/len(avg_train_loss)

    # control accuracy is how good on control images, AP is for non-control (positive) images
    wandb.log({"train loss - diff": avg_train_loss, "eval AP iou=0.5:0.95 - diff": iou, "eval AP iou=0.50 - diff": iou50, "eval AP iou=0.75 - diff": iou75, "control accuracy": control_accuracy})
    
    #if (step_idx * 100) % 10 == 0:
    #    torch.save(model, f"fcnn{step_idx}.pth")
    
    if do_step:
        train_loader.dataset.difficulty_step(num_add=NUM_ADD)

torch.save(model, f"fcnn_final.pth")