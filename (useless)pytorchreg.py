import json
from torchvision import transforms
import torch
from torchmetrics.detection import IntersectionOverUnion
import matplotlib.pyplot as plt 
from matplotlib import cm
from PIL import Image
import os
from tqdm import tqdm

transform = transforms.Compose([
    transforms.Resize(1024),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

device = torch.device("cuda")
cpu_device = torch.device("cpu")

PATH = "fcnn108.pth"

model = torch.load(PATH).to(device)

annotations = json.load(open(".Iirc /refineddataset/nodules.json"))
metadata = json.load(open("./refineddataset/nodulemetadata.json"))

metric = IntersectionOverUnion()

areas, brightnesses, ious = [], [], []

xray_path = "./refineddataset/testxrays"

for image in tqdm(os.listdir(xray_path)):
    nodules = annotations[image.split(".")[0]]
    image = Image.open(os.path.join(xray_path, image))
    image = transform(image).to(device)

    with torch.no_grad():
        outputs = model([image])

    outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

    for nodule_name, nodule_bbox in nodules:
        _, area, brightness = metadata[nodule_name]

        nodule_target = [{
            "boxes": torch.tensor([nodule_bbox], dtype=torch.float32),
            "labels": torch.ones(1, dtype=torch.int64),  # Assuming label for all boxes is 1 (nodule)
        }]

        iou = metric(preds=outputs, target=nodule_target)["iou"].item()

        if iou != 0:
            areas.append(area)
            brightnesses.append(brightness)
            ious.append(iou)


data = zip(areas, brightnesses)

#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#ax.scatter(areas, brightnesses, ious)

plt.scatter(brightnesses, ious)
plt.show()