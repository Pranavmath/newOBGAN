from datasets import NoduleDataset, CurriculumNoduleDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import random
import json

# does image need to norm to -1, 1 or 0, 1 for faster rcnn

def run():
    length = len(nd)

    num_negative = 0

    for _, target in tqdm(nd):
        # if negative
        if not list(target["boxes"]):
            num_negative += 1

    print(length)
    print(length * 0.2, num_negative)

nd = CurriculumNoduleDataset("./refineddataset/xrays", "./refineddataset/controlimages", "./refineddataset/nodules.json", "./refineddataset/fakedifficulties.json", 0.2, None)
nd.set_difficulty(0.01)


dl = DataLoader(nd, batch_size=1, shuffle=True)

print(len(dl))

dl.dataset.set_difficulty(0.05)

print(len(dl))
