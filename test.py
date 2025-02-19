import json

difficulties = json.load(open("./refineddataset/difficulties.json"))
difficulties = list(difficulties.values())

print(min(difficulties), max(difficulties))