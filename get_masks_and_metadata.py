import os
from PIL import Image
import json
from tqdm import tqdm
from utils import get_bounding_box, get_area, get_brightness, get_mask
import cv2
import numpy as np

SIZE = 256
# we make the bbox width and height a PAD value smaller so that SAM doesnt detect artificts at the border of the nodule 
PAD = 4

nodule_metadata = {}

subtracted_path = "./syntheticdataset/generatednodules"
mask_path = "./syntheticdataset/generated_masks"

i = 0

for nodule_name in tqdm(os.listdir(subtracted_path)):
    nodule = Image.open(os.path.join(subtracted_path, nodule_name))

    if nodule.getbbox() == None:
        i += 1
        continue

    xmin, ymin, xmax, ymax = nodule.getbbox()
    width, height = xmax-xmin, ymax-ymin

    """
    new_bbox = nodule_metadata[nodule_name][0]
    new_width, new_height = new_bbox[2] - new_bbox[0], new_bbox[3] - new_bbox[1]

    residuals.append(width-new_width)
    residuals.append(height-new_height)
    """
    
    width, height = width - PAD, height - PAD
    

    subtract = cv2.imread(os.path.join(subtracted_path, nodule_name), cv2.IMREAD_GRAYSCALE)
    assert subtract.shape == (SIZE, SIZE)
    
    bbox = [SIZE//2 - width//2, SIZE//2 - height//2, SIZE//2 + width//2, SIZE//2 + height//2] # xmin, ymin, xmax, ymax


    try:
        mask = get_mask(subtract, bbox)
    except ValueError:
        i += 1
        continue

    final_bbox = get_bounding_box(mask)
    w, h = final_bbox[2] - final_bbox[0], final_bbox[3] - final_bbox[1]

    if min(w, h) < 3:
        print("Too Small")
        i += 1
        continue

    area = get_area(mask)
    brightness = round(get_brightness(subtract, mask), 4)

    nodule_metadata[nodule_name] = [final_bbox, area, brightness]

    
    mask_img = Image.fromarray(mask * 255, "L")
    mask_img.save(os.path.join(mask_path, f"{nodule_name}"))


print(i)

with open("./syntheticdataset/nodulemetadata.json", "w") as f:
    json.dump(nodule_metadata, f, default=int)


"""
plt.hist(residuals, 100)

plt.show()
"""