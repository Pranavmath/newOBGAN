import os
import json
import numpy as np

diff = {}

metadata = json.load(open("./refineddataset/nodulemetadata.json"))
nodules = json.load(open("./refineddataset/nodules.json"))
xray_path = "./refineddataset/xrays"

areas, brightnesses = [area for _, area, _ in metadata.values()], [brightness for _, _, brightness in metadata.values()]
mean_area, std_area = np.mean(areas), np.std(areas)
mean_brightness, std_brightness = np.mean(brightnesses), np.std(brightnesses)

for xray in os.listdir(xray_path):
    diffs = []
    
    for nodule_name, _ in nodules[xray.split(".")[0]]:
        _, area, brightness = metadata[nodule_name]
        area = (area - mean_area) / std_area
        brightness = (brightness - mean_brightness) / std_brightness
        diff_value = -1 * (area + brightness) / 2
        diffs.append(diff_value)


    diff[xray] = sum(diffs)/len(diffs)

with open("./refineddataset/difficulties.json", "w") as f:
    json.dump(diff, f)