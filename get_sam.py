import os
from PIL import Image, ImageEnhance
from sam import get_mask
import json
from tqdm import tqdm
from utils import get_bounding_box, get_area, get_brightness
import numpy as np
import matplotlib.pyplot as plt

SIZE = 256
# we make the bbox width and height a PAD value smaller so that SAM doesnt detect artificts at the border of the nodule 
PAD = 4



subtracted_path = "./refineddataset/subtracts"
mask_path = "./refineddataset/sam_masks"

with open("./refineddataset/nodules.json") as f:
    nodule_dict = json.load(f)


with open("./refineddataset/nodulemetadata.json", "r") as f:
    nodule_metadata = json.load(f)

residuals = []

for img_name in tqdm(nodule_dict.keys()):
    for nodule_name, bbox in nodule_dict[img_name]:
        
        xmin, ymin, xmax, ymax = bbox
        width, height = xmax-xmin, ymax-ymin

        new_bbox = nodule_metadata[nodule_name][0]
        new_width, new_height = new_bbox[2] - new_bbox[0], new_bbox[3] - new_bbox[1]

        residuals.append(width-new_width)
        residuals.append(height-new_height)

        
        """
        width, height = width - PAD, height - PAD
        

        subtract = Image.open(os.path.join(subtracted_path, f"{nodule_name}.jpg")).convert("L")

        
        assert subtract.size == (SIZE, SIZE)
        
        bbox = [SIZE//2 - width//2, SIZE//2 - height//2, SIZE//2 + width//2, SIZE//2 + height//2] # xmin, ymin, xmax, ymax

        # we increase contrast so its easier for SAM to get mask
        mask = get_mask(ImageEnhance.Contrast(subtract).enhance(2), bbox)
        

        mask = np.array(Image.open(os.path.join(mask_path, f"{nodule_name}.jpg")).convert("1"))

        
        final_bbox = get_bounding_box(mask)
        area = get_area(mask)
        brightness = round(get_brightness(subtract, mask), 4)

        nodule_metadata[nodule_name] = [final_bbox, area, brightness]

        
        sam_mask_img = Image.fromarray(mask * 255, "L")
        sam_mask_img.save(os.path.join("./refineddataset/sam_masks", f"{nodule_name}.jpg"))
        """

plt.hist(residuals, 100)

plt.show()