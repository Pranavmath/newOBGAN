import os
from PIL import Image, ImageChops
from sam import get_mask
import json
from tqdm import tqdm
import numpy as np
from utils import get_bounding_box, get_area, get_brightness

SIZE = 256

nodules_path = "./refineddataset/nodules"
inpainted_path = "./refineddataset/inpainted"
subtracted_path = "./refineddataset/subtracted"

with open("./refineddataset/nodules.json") as f:
    nodule_dict = json.load(f)


for img_name in tqdm(nodule_dict.keys()):
    for nodule_name, bbox in nodule_dict[img_name]:
        xmin, ymin, xmax, ymax = bbox
        width, height = xmax-xmin, ymax-ymin
            
        nodule = Image.open(os.path.join(inpainted_path, f"{nodule_name}_truth.png")).convert("L")
        inpainted = Image.open(os.path.join(inpainted_path, f"{nodule_name}_out.png")).convert("L")

        subtract = ImageChops.subtract(nodule, inpainted).convert("L")

        assert nodule.size == inpainted.size == subtract.size == (SIZE, SIZE)
        
        bbox = [SIZE//2 - width//2, SIZE//2 - height//2, SIZE//2 + width//2, SIZE//2 + height//2] # xmin, ymin, xmax, ymax

        mask = get_mask(subtract, bbox)

        #final_bbox = get_bounding_box(mask)
        #area = get_area(mask)
        #brightness = get_brightness(nodule, mask)

        subtract.save(os.path.join(subtracted_path, f"{nodule_name}.jpg"))

        sam_mask_img = Image.fromarray(mask * 255, "L")

        sam_mask_img.save(os.path.join("./refineddataset/sam_masks", f"{nodule_name}.jpg"))
