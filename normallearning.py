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
from pytorchref.coco_eval import CocoEvaluator
from pytorchref.coco_utils import get_coco_api_from_dataset, _get_iou_types
import wandb

wandb.init(project="diff model training")

def collate_fn(batch):
    return tuple(zip(*batch))

def get_model():
    model = fcnn(weights="DEFAULT")
    num_classes = 2  # 1 class (nodule) + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


transform = transforms.Compose([
    transforms.Resize(1024),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

NUM_EPOCHS = 1000
# doing batch size of 4 since 1, 2, or 4 was recommended for faster rcnn
BATCH_SIZE = 4
SAVE_MODEL_INTERVAL = 12

device = torch.device("cuda")
cpu_device = torch.device("cpu")

model = get_model().to(device)

train_dataset = NoduleDataset("./refineddataset/trainxrays", "./refineddataset/control", "./refineddataset/nodules.json", 0.2, transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

val_dataset = NoduleDataset("./refineddataset/testxrays", "./refineddataset/control", "./refineddataset/nodules.json", 0.2, transform)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)


params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.Adam(
    params,
    lr=0.005,
    weight_decay=0.0005
)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

scaler = torch.amp.GradScaler()

wandb.watch(model)

print("Started Training")

for epoch in range(NUM_EPOCHS):
    print(f"Training for epoch: {epoch}")

    model.train()

    avg_train_loss = []

    for images, targets in train_loader:
        optimizer.zero_grad()

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        avg_train_loss.append(losses.item())


    print(f"Evaluating for epoch: {epoch}")

    model.eval()

    
    coco = get_coco_api_from_dataset(val_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in val_loader:
        images = list(image.to(device) for image in images)

        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}

        coco_evaluator.update(res)

    
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    # first - iou=0.5:0.95, second - iou=0.50, third - iou=0.75
    iou, iou50, iou75 = coco_evaluator.coco_eval["bbox"].stats[:3]
    avg_train_loss = sum(avg_train_loss)/len(avg_train_loss)

    wandb.log({"train loss - epoch": avg_train_loss, "eval AP iou=0.5:0.95 - epoch": iou, "eval AP iou=0.50 - epoch": iou50, "eval AP iou=0.75 - epoch": iou75})

    if epoch % SAVE_MODEL_INTERVAL == 0:
        torch.save(model, f"fcnn{epoch}.pth")
    


torch.save(model, f"fcnn_final.pth")







"""
def run():
    length = len(nd)

    num_negative = 0

    for _, target in tqdm(nd):
        # if negative
        if not list(target["boxes"]):
            num_negative += 1

    print(length, length * 0.2, num_negative)
"""
