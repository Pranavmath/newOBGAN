import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import json
from collections import deque
from typing import overload
import torchvision

# image norm to 0, 1 for faster rcnn
# image norm to 0, 1 for faster rcnn

IMAGE_SIZE = 1024

class NoduleDataset(Dataset):
    def __init__(self, xray_dir, control_dir, annotations_file, negative_sample_percentage=0.2, transform=None):
        """
        Args:
            xray_dir (str): Path to the directory containing xray images.
            control_dir (str): Path to the directory containing control (negative) images.
            annotations_file (str): Path to the JSON file containing annotations for nodule images.
            negative_sample_percentage (float): Percentage of data that is control image.
            transform (callable, optional): A function/transform to apply to the image.
        """
        self.xray_dir = xray_dir
        self.control_dir = control_dir
        self.annotations = json.load(open(annotations_file))
        self.annotations = {k + ".jpg": [annotation[1] for annotation in v] for k, v in self.annotations.items()}
        self.nodule_images = os.listdir(self.xray_dir)
        self.transform = transform

        positive_sample_percentage = 1 - negative_sample_percentage
        self.length = len(self.nodule_images) // positive_sample_percentage
        self.idx_is_negative = [(True, None) for _ in range(int(self.length * negative_sample_percentage))] + [(False, image) for image in self.nodule_images]
        random.shuffle(self.idx_is_negative)

        assert self.length == len(self.idx_is_negative)

    def __len__(self):
        return len(self.idx_is_negative)

    def __getitem__(self, idx):
        if self.idx_is_negative[idx][0]:
            # Return a control image (negative sample)
            control_image_name = random.choice(os.listdir(self.control_dir))
            control_image_path = os.path.join(self.control_dir, control_image_name)
            image = Image.open(control_image_path).convert("RGB")

            # Define the negative target
            boxes = torch.zeros((0, 4), dtype=torch.float32) 
            target = {
                "boxes": boxes,
                "labels": torch.zeros(0, dtype=torch.int64),
                "image_id": torch.tensor([idx], dtype=torch.int64),
                "area": torch.zeros(0, dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64)
            }

        else:
            # Return a nodule image (positive sample)
            image_name = self.idx_is_negative[idx][1]
            image_path = os.path.join(self.xray_dir, image_name)
            image = Image.open(image_path).convert("RGB")

            # Get the bounding boxes for the current nodule image
            boxes = torch.tensor(self.annotations[image_name], dtype=torch.float32)

            boxes = torchvision.tv_tensors.BoundingBoxes(boxes, format=torchvision.tv_tensors.BoundingBoxFormat("XYXY"), canvas_size=(IMAGE_SIZE, IMAGE_SIZE))
            boxes, labels = self.transform(boxes, labels)

            # Define the target for the positive sample
            target = {
                "boxes": boxes,
                "labels": torch.ones(len(boxes), dtype=torch.int64),  # Assuming label for all boxes is 1 (nodule)
                "image_id": torch.tensor([idx], dtype=torch.int64),
                "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64)
            }

        if self.transform:
            image = self.transform(image)

        return image, target



class CurriculumNoduleDataset(Dataset):
    def __init__(self, xray_dir, control_dir, annotations_file, difficulty_file, negative_sample_percentage=0.25, transform=None):
        """
        Args:
            xray_dir (str): Path to the directory containing xray images.
            control_dir (str): Path to the directory containing control (negative) images.
            annotations_file (str): Path to the JSON file containing annotations for nodule images.
            difficulty_file (str): Path to the JSON file containing difficulty levels for nodule images.
            negative_sample_percentage (float): Percentage of times to return a control image instead of a nodule image.
            transform (callable, optional): A function/transform to apply to the image.
        """
        self.xray_dir = xray_dir
        self.nodule_images = os.listdir(self.xray_dir)
        self.nodule_images = os.listdir(self.xray_dir)
        self.control_dir = control_dir
        self.annotations = json.load(open(annotations_file))
        self.annotations = {k + ".jpg": [annotation[1] for annotation in v] for k, v in self.annotations.items()}
        self.difficulties = json.load(open(difficulty_file))

        self.difficulties = {k: v for k, v in self.difficulties.items() if k in self.nodule_images}

        self.difficulties_deque = deque(sorted(self.difficulties.items(), key=lambda t: t[1]))

        self.difficulties = {k: v for k, v in self.difficulties.items() if k in self.nodule_images}

        self.difficulties_deque = deque(sorted(self.difficulties.items(), key=lambda t: t[1]))
        self.transform = transform
        self.current_difficulty = float("-inf")
        self.current_difficulty = float("-inf")

        self.negative_sample_percentage = negative_sample_percentage
        self.idx_is_negative = []

    def _update_idx_is_negative1(self, old_difficulty, new_difficulty):
        if new_difficulty < old_difficulty:
            raise ValueError("new_difficulty must be >= old_difficulty")

    def _update_idx_is_negative1(self, old_difficulty, new_difficulty):
        if new_difficulty < old_difficulty:
            raise ValueError("new_difficulty must be >= old_difficulty")

        for image in self.nodule_images:
            if old_difficulty < self.difficulties[image] <= new_difficulty:
                self.idx_is_negative.append((False, image))


        num_positive_images = 0
        for is_negative, _ in self.idx_is_negative:
            if not is_negative: num_positive_images += 1
        
        length = int(num_positive_images // (1 - self.negative_sample_percentage))
        num_negative_to_add = length - len(self.idx_is_negative)
        self.idx_is_negative += [(True, None) for _ in range(num_negative_to_add)]

        assert length == len(self.idx_is_negative)

        random.shuffle(self.idx_is_negative)
    
    def _update_idx_is_negative2(self, num_add):
        for _ in range(num_add):
            if self.difficulties_deque:
                image, diff = self.difficulties_deque.popleft()
                self.idx_is_negative.append((False, image))
        

        num_positive_images = 0
        for is_negative, _ in self.idx_is_negative:
            if not is_negative: num_positive_images += 1
        
        length = int(num_positive_images // (1 - self.negative_sample_percentage))
        num_negative_to_add = length - len(self.idx_is_negative)
        self.idx_is_negative += [(True, None) for _ in range(num_negative_to_add)]

        assert length == len(self.idx_is_negative)

        random.shuffle(self.idx_is_negative)
    
    def _update_idx_is_negative2(self, num_add):
        for _ in range(num_add):
            if self.difficulties_deque:
                image, diff = self.difficulties_deque.popleft()
                self.idx_is_negative.append((False, image))
        
        num_positive_images = 0
        for is_negative, _ in self.idx_is_negative:
            if not is_negative: num_positive_images += 1
        
        length = int(num_positive_images // (1 - self.negative_sample_percentage))
        num_negative_to_add = length - len(self.idx_is_negative)
        self.idx_is_negative += [(True, None) for _ in range(num_negative_to_add)]

        assert length == len(self.idx_is_negative)

        random.shuffle(self.idx_is_negative)


    def set_difficulty(self, difficulty):
        """Set the current difficulty level."""
        self._update_idx_is_negative1(self.current_difficulty, difficulty)
        self._update_idx_is_negative1(self.current_difficulty, difficulty)
        self.current_difficulty = difficulty
    
    def difficulty_step(self, num_add=10):
        self._update_idx_is_negative2(num_add=num_add)
    
    def difficulty_step(self, num_add=10):
        self._update_idx_is_negative2(num_add=num_add)


    def __len__(self):
        return len(self.idx_is_negative)


    def __getitem__(self, idx):
        if self.idx_is_negative[idx][0]:
            # Return a control image (negative sample)
            control_image_name = random.choice(os.listdir(self.control_dir))
            control_image_path = os.path.join(self.control_dir, control_image_name)
            image = Image.open(control_image_path).convert("RGB")
            boxes = torch.zeros((0, 4), dtype=torch.float32) 
            
            target = {
                "boxes": boxes,
                "labels": torch.zeros(0, dtype=torch.int64),
                "image_id": torch.tensor([idx], dtype=torch.int64),
                "area": torch.zeros(0, dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64)
            }

        else:
            image_name = self.idx_is_negative[idx][1]
            image_path = os.path.join(self.xray_dir, image_name)
            image = Image.open(image_path).convert("RGB")
            boxes = torch.tensor(self.annotations[image_name], dtype=torch.float32)

            boxes = torchvision.tv_tensors.BoundingBoxes(boxes, format=torchvision.tv_tensors.BoundingBoxFormat("XYXY"), canvas_size=(IMAGE_SIZE, IMAGE_SIZE))
            boxes, labels = self.transform(boxes, labels)


            target = {
                "boxes": boxes,
                "labels": torch.ones(len(boxes), dtype=torch.int64),     # Assuming label for all boxes is 1 (nodule)
                "image_id": torch.tensor([idx], dtype=torch.int64),
                "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64)
            }

        if self.transform:
            image = self.transform(image)

        return image, target
