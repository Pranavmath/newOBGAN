import numpy as np
import cv2
import numpy as np
from skimage.segmentation import flood_fill

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


def watershed_segmentation(gray):    
    # Apply thresholding to get a binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Remove noise by applying morphological operations
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Identify sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Identify sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
    # Identify unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Convert the image to color (BGR) so that Watershed can process it
    color_image = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
    
    # Label the markers
    _, markers = cv2.connectedComponents(sure_fg)
    
    # Add one to all labels to ensure the background is not labeled as 0
    markers = markers + 1
    
    # Mark the region of unknown with zero
    markers[unknown == 255] = 0
    
    # Apply the Watershed algorithm
    cv2.watershed(color_image, markers)
    
    # Mark boundaries in the original image
    color_image[markers == -1] = [255, 0, 0]  # Boundaries in red
    
    # Convert back to BGR for displaying in OpenCV
    segmented_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    
    return segmented_image, markers


def get_mask(image, bbox):
  xmin, ymin, xmax, ymax = bbox

  _, markers = watershed_segmentation(image)

  mask = np.ones_like(markers, dtype=np.uint8)
  mask[markers == -1] = 0

  size = len(mask)

  footprint = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

  mask = flood_fill(mask, (1, 1), 0, footprint=footprint)
  mask = flood_fill(mask, (1, size-2), 0, footprint=footprint)
  mask = flood_fill(mask, (size-2, 1), 0, footprint=footprint)
  mask = flood_fill(mask, (size-2, 254), 0, footprint=footprint)

  # ------------------------------

  cropped_mask = np.zeros_like(mask)
  cropped_mask[ymin:ymax, xmin:xmax] = mask[ymin:ymax, xmin:xmax]

  contours, _ = cv2.findContours(cropped_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  max_contour = max(contours, key=cv2.contourArea)

  cropped_mask = np.zeros_like(cropped_mask)
  cv2.drawContours(cropped_mask, [max_contour], -1, 1, thickness=cv2.FILLED)
  

  return cropped_mask