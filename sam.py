import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor

def get_mask(image, bbox):
    """
    Generate a single mask for the given image and bounding box using the Segment Anything Model (SAM).

    Args:
    - image (np.ndarray): Grayscale image of shape (256, 256).
    - bbox (tuple): Bounding box defined as (x_min, y_min, x_max, y_max).

    Returns:
    - mask (np.ndarray): Binary mask of the same size as the input image, with values 0 or 1.
    """
    # Load the pre-trained SAM model (adjust the path to the checkpoint as needed)
    sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
    predictor = SamPredictor(sam)
    
    image_tensor = np.repeat(np.array(image)[..., np.newaxis], 3, axis=2)
    
    # Set the image to the predictor
    predictor.set_image(image_tensor)
    
    sam_bbox = np.array(bbox)
    
    # Get the masks and their corresponding scores from SAM
    masks, scores, _ = predictor.predict(box=sam_bbox)
    
    # Select the mask with the highest confidence score
    best_mask_index = scores.argmax()
    best_mask = masks[best_mask_index]

    best_mask = best_mask.astype(np.uint8)

    # Convert mask to binary (0 or 1)
    #binary_mask = (best_mask > 0.5).astype(np.uint8)
    
    return best_mask

if __name__ == "__main__":
    bbox = [84, 73, 172, 183]
    mask = get_mask(np.array(Image.open("./refineddataset/nodules/000000.jpg")), bbox)
    Image.fromarray(mask * 255, "L").save("test.jpg")
