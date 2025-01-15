from PIL import Image
import numpy as np


def get_area(mask):
    """
    Returns the area of the mask where the value is 1.
    
    Parameters:
    mask (numpy.ndarray): A binary mask as a numpy array.
    
    Returns:
    int: The area (number of pixels) where the mask has a value of 1.
    """
    return np.sum(mask == 1)


def get_brightness(image, mask):
    """
    Returns the average brightness of the image in the region where the mask has a value of 1.
    
    Parameters:
    image (PIL.Image.Image): A grayscale PIL image.
    mask (numpy.ndarray): A binary mask as a numpy array.
    
    Returns:
    float: The average brightness in the masked region.
    """
    # Convert the PIL image to a numpy array
    image_array = np.array(image)
    
    # Apply the mask to the image array
    masked_image = image_array[mask == 1]
    
    return np.mean(masked_image)


def get_bounding_box(mask):
    """
    Get the bounding box for the region of interest (ROI) in a binary mask.

    Parameters:
    mask (numpy.ndarray): A numpy array where the region of interest is represented by 1s.

    Returns:
    A tuple (x_min, y_min, x_max, y_max) representing the coordinates of the bounding box
    """

    # Get the indices of the array where the value is 1
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    # Find the min and max row and column indices
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    # Return the bounding box as (x_min, y_min, x_max, y_max)
    return col_min, row_min, col_max, row_max