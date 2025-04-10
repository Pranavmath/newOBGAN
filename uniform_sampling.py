# generate testing and training sets so that it the samples in the test dataset are uniformally sampled across area and brightness

import json
import numpy as np
from PIL import Image
from utils import get_bounding_box, get_area, get_brightness
import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import shutil

def sample_points_kmeans_scaled(points, n):
    # Scale data to normalize variance
    scaler = StandardScaler()
    scaled_points = scaler.fit_transform(points)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n, random_state=42, init="k-means++", max_iter=300)
    kmeans.fit(scaled_points)
    
    # Transform centroids back to original scale
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    return centroids

# ----------------------------------------------------------------------

mask_path = "./refineddataset/generated_masks"
subtracted_path = "./refineddataset/subtracts"

with open("./refineddataset/nodules.json") as f:
    nodule_dict = json.load(f)

with open("./refineddataset/nodulemetadata.json") as f:
    nodulemetadata = json.load(f)

nodules = {}

for xray_name in tqdm(nodule_dict.keys()):
    for (nodule_name, _) in nodule_dict[xray_name]:
        subtract = cv2.imread(os.path.join(subtracted_path, f"{nodule_name}.jpg"), cv2.IMREAD_GRAYSCALE)

        mask = Image.open(os.path.join(mask_path, f"{nodule_name}.jpg")).convert("1")
        mask = np.array(mask, dtype=np.uint8)

        bbox, area, brightness = nodulemetadata[nodule_name]
        bbox = tuple(bbox)

        assert (bbox, area, brightness) == (get_bounding_box(mask), get_area(mask), round(get_brightness(subtract, mask), 4))

        nodules[(area, brightness)] = xray_name



# info: x - area, y - brightness
info = np.array(list(nodules.keys()))

sample = [np.argmin(np.linalg.norm(info - row, axis=1)) for row in sample_points_kmeans_scaled(info, n=200)]


xrays = set()

for idx in sample:
    t = tuple(info[idx])
    assert t in nodules.keys()
    xrays.add(nodules[t])


print(len(sample), len(xrays))

for xray in nodule_dict.keys():
    fname = f"{xray}.jpg"
    og_path = os.path.join("./refineddataset/xrays", fname)

    if xray in xrays:
        destination_path = os.path.join("./refineddataset/testxrays", fname)
    else:
        destination_path = os.path.join("./refineddataset/trainxrays", fname)

    shutil.copy(og_path, destination_path)



sample = np.array(sample, dtype=int)


new_info = np.delete(info, sample, axis=0)

print(len(info), len(sample), len(info)-len(sample), len(new_info))

plt.figure(1)
plt.scatter(info[:, 0], info[:, 1])

plt.figure(2)
plt.scatter(info[sample][:, 0], info[sample][:, 1])

plt.figure(3)
plt.scatter(new_info[:, 0], new_info[:, 1])

plt.title("Avg. Brightness and Area Scatterplot")
plt.xlabel("Area")
plt.ylabel("Avg. Brightness")


plt.show()
