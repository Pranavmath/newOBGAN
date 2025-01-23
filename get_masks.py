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


subtracted_path = "./refineddataset/subtracts"
mask_path = "./refineddataset/generated_masks"

with open("./refineddataset/nodules.json") as f:
    nodule_dict = json.load(f)


nodule_metadata = {}


#residuals = []

too_dark = []

i = 0

for img_name in tqdm(nodule_dict.keys()):
    for nodule_name, bbox in nodule_dict[img_name]:
        
        xmin, ymin, xmax, ymax = bbox
        width, height = xmax-xmin, ymax-ymin

        """
        new_bbox = nodule_metadata[nodule_name][0]
        new_width, new_height = new_bbox[2] - new_bbox[0], new_bbox[3] - new_bbox[1]

        residuals.append(width-new_width)
        residuals.append(height-new_height)
        """
        
        width, height = width - PAD, height - PAD
        

        subtract = cv2.imread(os.path.join(subtracted_path, f"{nodule_name}.jpg"), cv2.IMREAD_GRAYSCALE)
        assert subtract.shape == (SIZE, SIZE)
        
        bbox = [SIZE//2 - width//2, SIZE//2 - height//2, SIZE//2 + width//2, SIZE//2 + height//2] # xmin, ymin, xmax, ymax


        try:
            mask = get_mask(subtract, bbox)
        except ValueError:
            too_dark.append(nodule_name)
            mask = Image.open(os.path.join(mask_path, f"{nodule_name}.jpg")).convert("1")
            mask = np.array(mask, dtype=np.uint8)

        final_bbox = get_bounding_box(mask)
        area = get_area(mask)
        brightness = round(get_brightness(subtract, mask), 4)

        nodule_metadata[nodule_name] = [final_bbox, area, brightness]

        
        mask_img = Image.fromarray(mask * 255, "L")
        mask_img.save(os.path.join(mask_path, f"{nodule_name}.jpg"))

        i += 1

print(too_dark)
print(i)

with open("./refineddataset/nodulemetadata.json", "w") as f:
    json.dump(nodule_metadata, f, default=int)


"""
plt.hist(residuals, 100)

plt.show()
"""