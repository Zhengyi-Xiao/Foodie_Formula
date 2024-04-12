from fastapi import FastAPI, File, UploadFile, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from asyncio import create_task

import requests
import torch
import torch.nn as nn

import io
import base64
import pandas as pd
import numpy as np

from PIL import Image
from io import BytesIO
import time

from openai import OpenAI
import os
import openai
import json
import time
from torchvision import transforms


OPENAI_API_KEY = ""
openai.api_key = OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = openai.api_key
os.environ["TOKENIZERS_PARALLELISM"] = "false"
client = OpenAI()

device = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import io
import requests
from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pre-trained YOLOv5 model
yolov5model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

Segprocessor = SegformerImageProcessor.from_pretrained(
    "prem-timsina/segformer-b0-finetuned-food"
)
Segmodel = AutoModelForSemanticSegmentation.from_pretrained(
    "prem-timsina/segformer-b0-finetuned-food"
).to(device)
Segmodel.eval()


def extract_cropped_image_by_labels(original_image, pred_seg, labels):
    if isinstance(original_image, Image.Image):
        original_image = np.array(original_image)

    # Assuming 'pred_seg' is a numpy array with the same dimensions as 'original_image'
    # Identify pixels belonging to any of the specified labels
    mask = np.isin(pred_seg, labels)
    rows, cols = np.where(mask)

    # If no pixels with the specified labels are found, return None
    if len(rows) == 0 or len(cols) == 0:
        return None, None, 0

    # Determine the bounding box for the labeled objects
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)

    # Calculate the area of the cropped region and the original image
    cropped_area = (max_row - min_row + 1) * (max_col - min_col + 1)

    # Crop the mask to the bounding box to calculate the score
    cropped_mask = mask[min_row : max_row + 1, min_col : max_col + 1]

    # Calcuate the origional image size
    width, height = original_image.shape[1], original_image.shape[0]

    # Calculate the score as the number of true values in the cropped mask
    score = np.sum(cropped_mask) / (width * height)

    # Calculate the score as the sum of true values in the cropped mask
    # score = np.sum(cropped_mask) / (sum(labels) / len(labels) * width * height)

    # Crop the original image to the bounding box
    cropped_image = original_image[min_row : max_row + 1, min_col : max_col + 1]

    # Convert the cropped image back to a PIL Image for consistent output
    cropped_image = Image.fromarray(cropped_image)

    cropped_pred_seg = pred_seg[min_row : max_row + 1, min_col : max_col + 1]

    # Create a mask for pixels labeled 0 (to set them to white)
    zero_label_mask = cropped_pred_seg == 0

    # Make color white for pixels labeled with 0 in the cropped region
    whitened_cropped_image = np.copy(cropped_image)
    whitened_cropped_image[zero_label_mask] = [255, 255, 255]

    # Convert the modified cropped image back to a PIL Image for consistent output
    cropped_image_with_white = Image.fromarray(whitened_cropped_image.astype(np.uint8))

    return cropped_image, cropped_image_with_white, score

#every 100g of the food
#        "amountVitamins": mcg
#        "amountMinerals": mg,
#        "amountProtein": g
#        "amountCarbohydrates": g,
#        "amountFats": g,
ontology = {
    "candy": {
        "item": "candy",
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 0.3,
        "amountCalories": 394,
        "amountProtein": 0,
        "amountEnergy": 0,
        "ediblePart": 0,
        "amountCarbohydrates": 0,
        "amountFats": 0.2,
        "foodId": 0
    },
    "french fries": {
        "item": "french fries",
        "amountCalories": 365.0,
        "amountEnergy": 1305.5,
        "ediblePart": 100.0,
        "amountProtein": 4,
        "amountCarbohydrates": 48,
        "amountFats": 17,
        "amountVitamins": 0,
        "amountMinerals": 800,
        "foodId": 30.0,
        "isUnspecifiedFood": 1.0
    },
    "chocolate": {
        "item": "chocolate",
        "amountCalories": 546.0,
        "amountEnergy": 2284.5,
        "ediblePart": 100.0,
        "foodId": 68.0,
        "isUnspecifiedFood": 1.0,
        "amountVitamins": 0,
        "amountMinerals": 600,
        "amountProtein": 8.1,
        "amountCarbohydrates": 63,
        "amountFats": 31.5
    },
    "ice cream": {
        "item": "ice cream",
        "activeCategory": 1.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0.3,
        "amountMinerals": 500,
        "amountCalories": 273,
        "amountProtein": 4.6,
        "amountEnergy": 0,
        "ediblePart": 0,
        "amountCarbohydrates": 31,
        "amountFats": 15,
        "foodId": 0
    },
    "cake": {
        "item": "cake",
        "amountCalories": 391.0,
        "amountEnergy": 1000.0,
        "ediblePart": 100.0,
        "foodId": 71.0,
        "isUnspecifiedFood": 1.0,
        "amountVitamins": 0.4,
        "amountMinerals": 469,
        "amountProtein": 3,
        "amountCarbohydrates": 56,
        "amountFats": 18
    },
    "wine": {
        "item": "wine",
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 0,
        "amountCalories": 122,
        "amountProtein": 0.1,
        "amountEnergy": 0,
        "ediblePart": 0,
        "amountCarbohydrates": 3.8,
        "amountFats": 0,
        "foodId": 0
    },
    "egg": {
        "item": "egg",
        "amountCalories": 155.0,
        "amountEnergy": 648.6,
        "ediblePart": 100.0,
        "foodId": 13.0,
        "isUnspecifiedFood": 1.0,
        "amountVitamins": 2,
        "amountMinerals": 400,
        "amountProtein": 13,
        "amountCarbohydrates": 0.7,
        "amountFats": 9.5
    },
    "avocado": {
        "item": "avocado",
        "amountCalories": 160.0,
        "amountEnergy": 967.0,
        "amountWater": 64.0,
        "ediblePart": 76.0,
        "foodId": 999692.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 500,
        "amountProtein": 2,
        "amountCarbohydrates": 8.5,
        "amountFats": 15
    },
    "banana": {
        "item": "banana",
        "amountCalories": 89.0,
        "amountEnergy": 276.0,
        "amountWater": 76.8,
        "ediblePart": 65.0,
        "foodId": 406.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 400,
        "amountProtein": 1.1,
        "amountCarbohydrates": 23,
        "amountFats": 0.3
    },
    "mango": {
        "item": "mango",
        "amountCalories": 60.0,
        "amountEnergy": 238.0,
        "amountWater": 82.4,
        "ediblePart": 100.0,
        "foodId": 805867.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 200,
        "amountProtein": 0.8,
        "amountCarbohydrates": 15,
        "amountFats": 0.4
    },
    "kiwi": {
        "item": "kiwi",
        "amountCalories": 44.0,
        "amountEnergy": 184.0,
        "amountWater": 84.6,
        "ediblePart": 87.0,
        "foodId": 429.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 4,
        "amountProtein": 1.1,
        "amountCarbohydrates": 15,
        "amountFats": 0.5
    },
    "watermelon": {
        "item": "watermelon",
        "amountCalories": 30.0,
        "amountEnergy": 63.0,
        "amountWater": 95.3,
        "ediblePart": 52.0,
        "foodId": 408.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 200,
        "amountProtein": 0.6,
        "amountCarbohydrates": 7.6,
        "amountFats": 0.2
    },
    "sausage": {
        "item": "sausage",
        "amountCalories": 336.0,
        "amountEnergy": 1405.9,
        "ediblePart": 100.0,
        "foodId": 62.0,
        "isUnspecifiedFood": 1.0,
        "amountVitamins": 1.1,
        "amountMinerals": 500,
        "amountProtein": 12,
        "amountCarbohydrates": 0.9,
        "amountFats": 28
    },
    "lamb": {
        "item": "lamb",
        "amountCalories": 294.0,
        "amountEnergy": 678.0,
        "amountWater": 70.1,
        "ediblePart": 83.0,
        "foodId": 1060.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0.1,
        "amountMinerals": 500,
        "amountProtein": 25,
        "amountCarbohydrates": 0,
        "amountFats": 21
    },
    "crab": {
        "item": "crab",
        "amountCalories": 87.0,
        "amountEnergy": 364.0,
        "amountWater": 79.0,
        "ediblePart": 35.0,
        "foodId": 700418.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 600,
        "amountProtein": 18,
        "amountCarbohydrates": 0,
        "amountFats": 0.7
    },
    "fish": {
        "item": "fish",
        "activeCategory": 1.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 3.7,
        "amountMinerals": 400,
        "amountCalories": 128,
        "amountProtein": 26,
        "amountEnergy": 0,
        "ediblePart": 0,
        "amountCarbohydrates": 0,
        "amountFats": 2.7,
        "foodId": 0
    },
    "shellfish": {
        "item": "shellfish",
        "isUnspecifiedFood": 0,
        "amountVitamins": 0.2,
        "amountMinerals": 500,
        "amountCalories": 188,
        "amountProtein": 8.5,
        "amountEnergy": 0,
        "ediblePart": 0,
        "amountCarbohydrates": 18,
        "amountFats": 8.5,
        "foodId": 0
    },
    "shrimp": {
        "item": "shrimp",
        "amountCalories": 71.0,
        "amountEnergy": 297.0,
        "amountWater": 80.1,
        "ediblePart": 45.0,
        "foodId": 1309.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0.2,
        "amountMinerals": 520,
        "amountProtein": 22,
        "amountCarbohydrates": 220,
        "amountFats": 1.8
    },
    "bread": {
        "item": "bread",
        "activeCategory": 1.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 786,
        "amountCalories": 308,
        "amountProtein": 10.4,
        "amountEnergy": 0,
        "ediblePart": 0,
        "amountCarbohydrates": 56,
        "amountFats": 3.9,
        "foodId": 0
    },
    "corn": {
        "item": "corn",
        "amountCalories": 99.0,
        "amountEnergy": 890.0,
        "amountWater": 6.5,
        "ediblePart": 100.0,
        "foodId": 3.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 325,
        "amountProtein": 3.5,
        "amountCarbohydrates": 22,
        "amountFats": 1.5
    },
    "pasta": {
        "item": "pasta",
        "contains": "WheatFour",
        "activeCategory": 1.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 200,
        "amountCalories": 196,
        "amountProtein": 7.2,
        "amountEnergy": 0,
        "ediblePart": 0,
        "amountCarbohydrates": 38,
        "amountFats": 1.2,
        "foodId": 0
    },
    "rice": {
        "item": "rice",
        "activeCategory": 1.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 50,
        "amountCalories": 133,
        "amountProtein": 2.8,
        "amountEnergy": 0,
        "ediblePart": 0,
        "amountCarbohydrates": 0,
        "amountFats": 0.3,
        "foodId": 0
    },
    "tofu": {
        "item": "tofu",
        "amountCalories": 83.0,
        "amountEnergy": 318.0,
        "amountWater": 84.6,
        "ediblePart": 100.0,
        "foodId": 888002.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 289,
        "amountProtein": 10,
        "amountCarbohydrates": 1.2,
        "amountFats": 5.3
    },
    "eggplant": {
        "item": "eggplant",
        "amountCalories": 35.0,
        "amountEnergy": 63.0,
        "amountWater": 92.7,
        "ediblePart": 92.0,
        "foodId": 334.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 189,
        "amountProtein": 0.8,
        "amountCarbohydrates": 8.7,
        "amountFats": 0.2
    },
    "cauliflower": {
        "item": "cauliflower",
        "amountCalories": 22.0,
        "amountEnergy": 105.0,
        "amountWater": 90.5,
        "ediblePart": 66.0,
        "foodId": 313.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 163,
        "amountProtein": 1.8,
        "amountCarbohydrates": 4.1,
        "amountFats": 0.5
    },
    "rape": {
        "item": "rape",
        "amountCalories": 18.0,
        "amountEnergy": 75.0,
        "amountWater": 93.3,
        "ediblePart": 69.0,
        "foodId": 341.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 222,
        "amountProtein": 0.8,
        "amountCarbohydrates": 2,
        "amountFats": 0.1
    },
    "lettuce": {
        "item": "lettuce",
        "amountCalories": 19.0,
        "amountEnergy": 79.0,
        "amountWater": 94.3,
        "ediblePart": 80.0,
        "foodId": 331.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 323,
        "amountProtein": 1.2,
        "amountCarbohydrates": 3.1,
        "amountFats": 0.3
    },
    "cucumber": {
        "item": "cucumber",
        "amountCalories": 14.0,
        "amountEnergy": 59.0,
        "amountWater": 96.5,
        "ediblePart": 77.0,
        "foodId": 318.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 160,
        "amountProtein": 0.7,
        "amountCarbohydrates": 3.7,
        "amountFats": 0.1
    },
    "carrot": {
        "item": "carrot",
        "amountCalories": 33.0,
        "amountEnergy": 138.0,
        "amountWater": 91.6,
        "ediblePart": 95.0,
        "foodId": 312.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 389,
        "amountProtein": 0.7,
        "amountCarbohydrates": 7.6,
        "amountFats": 0.2
    },
    "broccoli": {
        "item": "broccoli",
        "amountCalories": 35.0,
        "amountEnergy": 113.0,
        "amountWater": 90.3,
        "ediblePart": 51.0,
        "foodId": 309.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 442,
        "amountProtein": 2.4,
        "amountCarbohydrates": 7.2,
        "amountFats": 0.4
    },
    "onion": {
        "item": "onion",
        "amountCalories": 44.0,
        "amountEnergy": 109.0,
        "amountWater": 92.1,
        "ediblePart": 83.0,
        "foodId": 322.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 186,
        "amountProtein": 1.4,
        "amountCarbohydrates": 10,
        "amountFats": 0.2
    },
    "green beans": {
        "item": "green beans",
        "amountCalories": 35.0,
        "amountEnergy": 71.0,
        "amountWater": 90.5,
        "ediblePart": 95.0,
        "foodId": 101.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 186,
        "amountProtein": 1.9,
        "amountCarbohydrates": 7.9,
        "amountFats": 0.3
    },
    "salad": {
        "item": "salad",
        "amountCalories": 24.0,
        "amountEnergy": 61.0,
        "amountWater": 92.6,
        "ediblePart": 81.0,
        "foodId": 9001.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 423,
        "amountProtein": 1.5,
        "amountCarbohydrates": 5,
        "amountFats": 0.2
    },
    "biscuit": {
        "item": "biscuit",
        "amountCalories": 429.0,
        "amountEnergy": 1795.0,
        "ediblePart": 100.0,
        "foodId": 17.0,
        "isUnspecifiedFood": 1.0,
        "amountVitamins": 0,
        "amountMinerals": 835,
        "amountProtein": 7,
        "amountCarbohydrates": 45,
        "amountFats": 16
    },
    "apple": {
        "item": "apple",
        "amountCalories": 52.0,
        "amountEnergy": 188.0,
        "amountWater": 85.6,
        "ediblePart": 94.0,
        "foodId": 419.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 108,
        "amountProtein": 0.3,
        "amountCarbohydrates": 14,
        "amountFats": 0.2
    },
    "apricot": {
        "item": "apricot",
        "amountCalories": 48.0,
        "amountEnergy": 117.0,
        "amountWater": 86.3,
        "ediblePart": 94.0,
        "foodId": 401.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 5,
        "amountMinerals": 286,
        "amountProtein": 1.4,
        "amountCarbohydrates": 11,
        "amountFats": 0.4
    },
    "lemon": {
        "item": "lemon",
        "amountCalories": 29.0,
        "amountEnergy": 46.0,
        "amountWater": 89.5,
        "ediblePart": 64.0,
        "foodId": 413.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 4,
        "amountMinerals": 159,
        "amountProtein": 1.1,
        "amountCarbohydrates": 9.3,
        "amountFats": 0.3
    },
    "pear": {
        "item": "pear",
        "amountCalories": 57.0,
        "amountEnergy": 172.0,
        "amountWater": 85.2,
        "ediblePart": 91.0,
        "foodId": 424.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 4,
        "amountMinerals": 126,
        "amountProtein": 0.4,
        "amountCarbohydrates": 15,
        "amountFats": 0.1
    },
    "fig": {
        "item": "fig",
        "amountCalories": 74.0,
        "amountEnergy": 197.0,
        "amountWater": 81.9,
        "ediblePart": 75.0,
        "foodId": 409.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 2,
        "amountMinerals": 332,
        "amountProtein": 0.8,
        "amountCarbohydrates": 19,
        "amountFats": 0.3
    },
    "pineapple": {
        "item": "pineapple",
        "amountCalories": 50.0,
        "amountEnergy": 167.0,
        "amountWater": 86.4,
        "ediblePart": 57.0,
        "foodId": 403.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 6,
        "amountMinerals": 120,
        "amountProtein": 0.5,
        "amountCarbohydrates": 13,
        "amountFats": 0.1
    },
    "grape": {
        "item": "grape",
        "amountCalories": 61.0,
        "amountEnergy": 255.0,
        "amountWater": 80.3,
        "ediblePart": 94.0,
        "foodId": 428.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 3.6,
        "amountMinerals": 121,
        "amountProtein": 0.7,
        "amountCarbohydrates": 18,
        "amountFats": 0.2
    },
    "melon": {
        "item": "melon",
        "amountCalories": 33.0,
        "amountEnergy": 138.0,
        "amountWater": 90.1,
        "ediblePart": 47.0,
        "foodId": 421.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 8.14,
        "amountMinerals": 304,
        "amountProtein": 0.8,
        "amountCarbohydrates": 8.2,
        "amountFats": 0.2
    },
    "orange": {
        "item": "orange",
        "amountCalories": 49.0,
        "amountEnergy": 142.0,
        "amountWater": 87.2,
        "ediblePart": 80.0,
        "foodId": 404.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 82.7,
        "amountMinerals": 233,
        "amountProtein": 0.9,
        "amountCarbohydrates": 13,
        "amountFats": 0.2
    },
    "soup": {
        "item": "soup",
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 50,
        "amountCalories": 24,
        "amountProtein": 1.2,
        "amountEnergy": 0,
        "ediblePart": 0,
        "amountCarbohydrates": 3,
        "amountFats": 0.8,
        "foodId": 0
    },
    "potato": {
        "item": "potato",
        "amountCalories": 110.0,
        "amountEnergy": 460.3,
        "ediblePart": 100.0,
        "foodId": 8.0,
        "isUnspecifiedFood": 1.0,
        "amountVitamins": 12,
        "amountMinerals": 600,
        "amountProtein": 2.5,
        "amountCarbohydrates": 21,
        "amountFats": 0.1
    },
    "strawberry": {
        "item": "strawberry",
        "amountCalories": 32.0,
        "amountEnergy": 113.0,
        "amountWater": 90.5,
        "ediblePart": 94.0,
        "foodId": 411.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 89,
        "amountMinerals": 165,
        "amountProtein": 0.7,
        "amountCarbohydrates": 7.7,
        "amountFats": 0.3
    },
    "cherry": {
        "item": "cherry",
        "amountCalories": 63.0,
        "amountEnergy": 159.0,
        "amountWater": 86.2,
        "ediblePart": 86.0,
        "foodId": 407.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 10,
        "amountMinerals": 222,
        "amountProtein": 1.1,
        "amountCarbohydrates": 16,
        "amountFats": 0.2
    },
    "blueberry": {
        "item": "blueberry",
        "amountCalories": 30.0,
        "amountEnergy": 126.0,
        "amountWater": 85.9,
        "ediblePart": 98.0,
        "foodId": 500021.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 32,
        "amountMinerals": 99,
        "amountProtein": 0.7,
        "amountCarbohydrates": 14,
        "amountFats": 0.3
    },
    "raspberry": {
        "item": "raspberry",
        "amountCalories": 34.0,
        "amountEnergy": 142.0,
        "amountWater": 84.6,
        "ediblePart": 100.0,
        "foodId": 412.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 25,
        "amountMinerals": 176,
        "amountProtein": 1.2,
        "amountCarbohydrates": 12,
        "amountFats": 0.7
    },
    "egg tart": {
        "item": "egg tart",
        "amountCalories": 217.0,
        "amountEnergy": 908.0,
        "ediblePart": 100.0,
        "foodId": 90.0,
        "isUnspecifiedFood": 1.0,
        "amountVitamins": 0.8,
        "amountMinerals": 120,
        "amountProtein": 6.1,
        "amountCarbohydrates": 29,
        "amountFats": 13
    },
    "popcorn": {
        "item": "popcorn",
        "amountCalories": 557.0,
        "amountEnergy": 1602.0,
        "amountWater": 4.0,
        "ediblePart": 100.0,
        "foodId": 3009.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 888,
        "amountProtein": 7.5,
        "amountCarbohydrates": 55,
        "amountFats": 34
    },
    "pudding": {
        "item": "pudding",
        "amountCalories": 133.0,
        "amountEnergy": 556.0,
        "amountWater": 69.3,
        "ediblePart": 100.0,
        "foodId": 800188.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 132,
        "amountProtein": 1.5,
        "amountCarbohydrates": 23,
        "amountFats": 3.8
    },
    "cheese butter": {
        "item": "cheese butter",
        "amountCalories": 269.0,
        "amountEnergy": 3171.0,
        "amountWater": 14.1,
        "ediblePart": 100.0,
        "foodId": 1900.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0.1,
        "amountMinerals": 323,
        "amountProtein": 15,
        "amountCarbohydrates": 20,
        "amountFats": 14
    },
    "coffee": {
        "item": "coffee",
        "amountCalories": 1.0,
        "amountEnergy": 8.0,
        "amountWater": 99.3,
        "ediblePart": 100.0,
        "foodId": 9030.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 2,
        "amountProtein": 0.1,
        "amountCarbohydrates": 0,
        "amountFats": 0
    },
    "juice": {
        "item": "juice",
        "amountCalories": 54.0,
        "amountEnergy": 774.0,
        "amountWater": 41.6,
        "ediblePart": 100.0,
        "foodId": 8026.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 2,
        "amountMinerals": 133,
        "amountProtein": 0.1,
        "amountCarbohydrates": 13,
        "amountFats": 0
    },
    "milk": {
        "item": "milk",
        "amountCalories": 46.0,
        "amountEnergy": 192.5,
        "ediblePart": 100.0,
        "foodId": 9.0,
        "isUnspecifiedFood": 1.0,
        "amountVitamins": 1.2,
        "amountMinerals": 160,
        "amountProtein": 3.3,
        "amountCarbohydrates": 4.8,
        "amountFats": 2
    },
    "tea": {
        "item": "tea",
        "amountCalories": 1.0,
        "amountEnergy": 0.0,
        "amountWater": 99.5,
        "ediblePart": 100.0,
        "foodId": 700459.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 0,
        "amountProtein": 0,
        "amountCarbohydrates": 0.3,
        "amountFats": 0
    },
    "almond": {
        "item": "almond",
        "amountCalories": 598.0,
        "amountEnergy": 1954.0,
        "amountWater": 6.1,
        "ediblePart": 100.0,
        "foodId": 900503.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 999,
        "amountProtein": 21,
        "amountCarbohydrates": 21,
        "amountFats": 53
    },
    "red beans": {
        "item": "red beans",
        "amountCalories": 127.0,
        "amountEnergy": 435.0,
        "amountWater": 62.3,
        "ediblePart": 41.0,
        "foodId": 100.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 545,
        "amountProtein": 8.7,
        "amountCarbohydrates": 23,
        "amountFats": 0.5
    },
    "soy": {
        "item": "soy",
        "amountCalories": 148.0,
        "amountEnergy": 1761.0,
        "amountWater": 0.0,
        "ediblePart": 100.0,
        "foodId": 910.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 455,
        "amountProtein": 15.5,
        "amountCarbohydrates": 7,
        "amountFats": 7.5
    },
    "walnut": {
        "item": "walnut",
        "amountCalories": 660.0,
        "amountEnergy": 2761.0,
        "amountWater": 6.3,
        "ediblePart": 39.0,
        "foodId": 510.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 547,
        "amountProtein": 15.5,
        "amountCarbohydrates": 14,
        "amountFats": 65
    },
    "peanut": {
        "item": "peanut",
        "amountCalories": 590.0,
        "amountEnergy": 2389.0,
        "amountWater": 7.1,
        "ediblePart": 79.0,
        "foodId": 501.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 735,
        "amountProtein": 24,
        "amountCarbohydrates": 21,
        "amountFats": 50
    },
    "date": {
        "item": "date",
        "amountCalories": 282.0,
        "amountEnergy": 519.0,
        "amountWater": 60.7,
        "ediblePart": 100.0,
        "foodId": 606531.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 765,
        "amountProtein": 2.5,
        "amountCarbohydrates": 75,
        "amountFats": 0.4
    },
    "olives": {
        "item": "olives",
        "amountCalories": 115.0,
        "amountEnergy": 594.0,
        "amountWater": 76.8,
        "ediblePart": 84.0,
        "foodId": 590.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 98,
        "amountProtein": 0.8,
        "amountCarbohydrates": 6.3,
        "amountFats": 11
    },
    "peach": {
        "item": "peach",
        "amountCalories": 39.0,
        "amountEnergy": 113.0,
        "amountWater": 90.7,
        "ediblePart": 91.0,
        "foodId": 425.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 196,
        "amountProtein": 0.9,
        "amountCarbohydrates": 9.5,
        "amountFats": 0.3
    },
    "steak": {
        "item": "steak",
        "amountCalories": 278.0,
        "amountEnergy": 588.0,
        "amountWater": 71.6,
        "ediblePart": 100.0,
        "foodId": 8653.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 328,
        "amountProtein": 26,
        "amountCarbohydrates": 0,
        "amountFats": 18
    },
    "pork": {
        "item": "pork",
        "amountCalories": 238.0,
        "amountEnergy": 2766.0,
        "amountWater": 19.3,
        "ediblePart": 100.0,
        "foodId": 1903.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0.9,
        "amountMinerals": 432,
        "amountProtein": 26,
        "amountCarbohydrates": 0,
        "amountFats": 14
    },
    "chicken duck": {
        "item": "chicken duck",
        "amountCalories": 220.0,
        "amountEnergy": 732.0,
        "amountWater": 68.7,
        "ediblePart": 68.0,
        "foodId": 1098.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 260,
        "amountProtein": 24,
        "amountCarbohydrates": 0.1,
        "amountFats": 13
    },
    "fried meat": {
        "item": "fried meat",
        "activeCategory": 1.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 0,
        "amountCalories": 226,
        "amountProtein": 28,
        "amountEnergy": 0,
        "ediblePart": 0,
        "amountCarbohydrates": 0,
        "amountFats": 12,
        "foodId": 0
    },
    "pizza": {
        "item": "pizza",
        "amountCalories": 271.0,
        "amountEnergy": 1134.0,
        "amountWater": 39.3,
        "ediblePart": 100.0,
        "foodId": 4.0,
        "isUnspecifiedFood": 1.0,
        "amountVitamins": 0,
        "amountMinerals": 0,
        "amountProtein": 0,
        "amountCarbohydrates": 0,
        "amountFats": 0
    },
    "wonton dumplings": {
        "item": "wonton dumplings",
        "amountCalories": 225.0,
        "amountEnergy": 941.4,
        "ediblePart": 100.0,
        "foodId": 11.0,
        "isUnspecifiedFood": 1.0,
        "amountVitamins": 0,
        "amountMinerals": 0,
        "amountProtein": 0,
        "amountCarbohydrates": 0,
        "amountFats": 0
    },
    "noodles": {
        "item": "noodles",
        "amountCalories": 309.0,
        "amountEnergy": 1293.0,
        "amountWater": 25.0,
        "ediblePart": 100.0,
        "foodId": 900018.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 0,
        "amountProtein": 0,
        "amountCarbohydrates": 0,
        "amountFats": 0
    },
    "pie": {
        "item": "pie",
        "amountCalories": 271.0,
        "amountEnergy": 1134.0,
        "amountWater": 39.3,
        "ediblePart": 100.0,
        "foodId": 4.0,
        "isUnspecifiedFood": 1.0,
        "amountVitamins": 0,
        "amountMinerals": 0,
        "amountProtein": 0,
        "amountCarbohydrates": 0,
        "amountFats": 0
    },
    "garlic": {
        "item": "garlic",
        "amountCalories": 41.0,
        "amountEnergy": 172.0,
        "amountWater": 80.0,
        "ediblePart": 75.0,
        "foodId": 301.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 0,
        "amountProtein": 0,
        "amountCarbohydrates": 0,
        "amountFats": 0
    },
    "tomato": {
        "item": "tomato",
        "amountCalories": 17.0,
        "amountEnergy": 71.0,
        "amountWater": 94.2,
        "ediblePart": 100.0,
        "foodId": 391.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 0,
        "amountProtein": 0,
        "amountCarbohydrates": 0,
        "amountFats": 0
    },
    "spring onion": {
        "item": "spring onion",
        "amountCalories": 26.0,
        "amountEnergy": 109.0,
        "amountWater": 92.1,
        "ediblePart": 83.0,
        "foodId": 322.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 0,
        "amountProtein": 0,
        "amountCarbohydrates": 0,
        "amountFats": 0
    },
    "pumpkin": {
        "item": "pumpkin",
        "amountCalories": 18.0,
        "amountEnergy": 75.0,
        "amountWater": 94.6,
        "ediblePart": 81.0,
        "foodId": 347.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 0,
        "amountProtein": 0,
        "amountCarbohydrates": 0,
        "amountFats": 0
    },
    "white radish": {
        "item": "white radish",
        "amountCalories": 11.0,
        "amountEnergy": 46.0,
        "amountWater": 95.6,
        "ediblePart": 99.0,
        "foodId": 342.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 0,
        "amountProtein": 0,
        "amountCarbohydrates": 0,
        "amountFats": 0
    },
    "asparagus": {
        "item": "asparagus",
        "amountCalories": 29.0,
        "amountEnergy": 121.0,
        "amountWater": 91.4,
        "ediblePart": 87.0,
        "foodId": 304.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 0,
        "amountProtein": 0,
        "amountCarbohydrates": 0,
        "amountFats": 0
    },
    "celery stick": {
        "item": "celery stick",
        "amountCalories": 20.0,
        "amountEnergy": 84.0,
        "amountWater": 88.3,
        "ediblePart": 80.0,
        "foodId": 343.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 0,
        "amountProtein": 0,
        "amountCarbohydrates": 0,
        "amountFats": 0
    },
    "cilantro mint": {
        "item": "cilantro mint",
        "amountCalories": 393.0,
        "amountEnergy": 1644.0,
        "amountWater": 0.2,
        "ediblePart": 100.0,
        "foodId": 19012.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 0,
        "amountProtein": 0,
        "amountCarbohydrates": 0,
        "amountFats": 0
    },
    "snow peas": {
        "item": "snow peas",
        "amountCalories": 76.0,
        "amountEnergy": 318.0,
        "amountWater": 76.1,
        "ediblePart": 47.0,
        "foodId": 103.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 0,
        "amountProtein": 0,
        "amountCarbohydrates": 0,
        "amountFats": 0
    },
    "bean sprouts": {
        "item": "bean sprouts",
        "amountCalories": 49.0,
        "amountEnergy": 205.0,
        "amountWater": 86.3,
        "ediblePart": 98.0,
        "foodId": 350.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 0,
        "amountProtein": 0,
        "amountCarbohydrates": 0,
        "amountFats": 0
    },
    "pepper": {
        "item": "pepper",
        "amountCalories": 25.0,
        "amountEnergy": 105.0,
        "amountWater": 87.8,
        "ediblePart": 89.0,
        "foodId": 335.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 0,
        "amountProtein": 0,
        "amountCarbohydrates": 0,
        "amountFats": 0
    },
    "French beans": {
        "item": "French beans",
        "amountCalories": 104.0,
        "amountEnergy": 435.0,
        "amountWater": 62.3,
        "ediblePart": 41.0,
        "foodId": 100.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 0,
        "amountProtein": 0,
        "amountCarbohydrates": 0,
        "amountFats": 0
    },
    "king oyster mushroom": {
        "item": "king oyster mushroom",
        "amountCalories": 69.0,
        "amountEnergy": 289.0,
        "amountWater": 85.7,
        "ediblePart": 12.0,
        "foodId": 1314.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 0,
        "amountProtein": 0,
        "amountCarbohydrates": 0,
        "amountFats": 0
    },
    "shiitake": {
        "item": "shiitake",
        "amountCalories": 296.0,
        "amountEnergy": 1238.0,
        "amountWater": 9.5,
        "ediblePart": 100.0,
        "foodId": 8036.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 0,
        "amountProtein": 0,
        "amountCarbohydrates": 0,
        "amountFats": 0
    },
    "enoki mushroom": {
        "item": "enoki mushroom",
        "amountCalories": 20.0,
        "amountEnergy": 84.0,
        "amountWater": 92.1,
        "ediblePart": 90.0,
        "foodId": 329.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 0,
        "amountProtein": 0,
        "amountCarbohydrates": 0,
        "amountFats": 0
    },
    "oyster mushroom": {
        "item": "oyster mushroom",
        "amountCalories": 20.0,
        "amountEnergy": 84.0,
        "amountWater": 92.1,
        "ediblePart": 90.0,
        "foodId": 329.0,
        "isUnspecifiedFood": 0,
        "amountVitamins": 0,
        "amountMinerals": 0,
        "amountProtein": 0,
        "amountCarbohydrates": 0,
        "amountFats": 0
    }
}

id2labels = {
    "0": "background",
    "1": "candy",
    "2": "egg tart",
    "3": "french fries",
    "4": "chocolate",
    "5": "biscuit",
    "6": "popcorn",
    "7": "pudding",
    "8": "ice cream",
    "9": "cheese butter",
    "10": "cake",
    "11": "wine",
    "12": "milkshake",
    "13": "coffee",
    "14": "juice",
    "15": "milk",
    "16": "tea",
    "17": "almond",
    "18": "red beans",
    "19": "cashew",
    "20": "dried cranberries",
    "21": "soy",
    "22": "walnut",
    "23": "peanut",
    "24": "egg",
    "25": "apple",
    "26": "date",
    "27": "apricot",
    "28": "avocado",
    "29": "banana",
    "30": "strawberry",
    "31": "cherry",
    "32": "blueberry",
    "33": "raspberry",
    "34": "mango",
    "35": "olives",
    "36": "peach",
    "37": "lemon",
    "38": "pear",
    "39": "fig",
    "40": "pineapple",
    "41": "grape",
    "42": "kiwi",
    "43": "melon",
    "44": "orange",
    "45": "watermelon",
    "46": "steak",
    "47": "pork",
    "48": "chicken duck",
    "49": "sausage",
    "50": "fried meat",
    "51": "lamb",
    "52": "sauce",
    "53": "crab",
    "54": "fish",
    "55": "shellfish",
    "56": "shrimp",
    "57": "soup",
    "58": "bread",
    "59": "corn",
    "60": "hamburg",
    "61": "pizza",
    "62": " hanamaki baozi",
    "63": "wonton dumplings",
    "64": "pasta",
    "65": "noodles",
    "66": "rice",
    "67": "pie",
    "68": "tofu",
    "69": "eggplant",
    "70": "potato",
    "71": "garlic",
    "72": "cauliflower",
    "73": "tomato",
    "74": "kelp",
    "75": "seaweed",
    "76": "spring onion",
    "77": "rape",
    "78": "ginger",
    "79": "okra",
    "80": "lettuce",
    "81": "pumpkin",
    "82": "cucumber",
    "83": "white radish",
    "84": "carrot",
    "85": "asparagus",
    "86": "bamboo shoots",
    "87": "broccoli",
    "88": "celery stick",
    "89": "cilantro mint",
    "90": "snow peas",
    "91": " cabbage",
    "92": "bean sprouts",
    "93": "onion",
    "94": "pepper",
    "95": "green beans",
    "96": "French beans",
    "97": "king oyster mushroom",
    "98": "shiitake",
    "99": "enoki mushroom",
    "100": "oyster mushroom",
    "101": "white button mushroom",
    "102": "salad",
    "103": "other ingredients",
}

async def generate_nutrient_analysis(responses):
    content = "Here is a list of food items and their nutrient information:\n" \
          f"{responses}\n\n" \
          "Please provide a concise and user-friendly nutrient analysis summary based on the given food items. " \
          "Include the following in your summary:\n" \
          "1. Key nutritional strengths and weaknesses of the food items\n" \
          "2. Comparison of their nutrient profile to the USDA dietary guidelines for a balanced diet\n" \
          "3. 1-2 actionable suggestions to improve the nutritional balance, focusing on the most important changes that can be easily implemented\n\n" \
          "Provide a clear, easy-to-understand, and engaging summary that can be displayed on a phone. Use a friendly tone and limit your response to 65 words."

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a knowledgeable nutrition expert. Provide a comprehensive, user-friendly, and engaging nutrient analysis summary based on the given context."
            },
            {
                "role": "user",
                "content": content
            }
        ]
    )

    final_summary = response.choices[0].message.content
    return final_summary


@app.post("/uploadImage")
async def upload_image(image: UploadFile = File(...)):
  
    # img_path = "/Users/zhengyi-xiao/Desktop/WechatIMG261.jpg"
    # image = Image.open(img_path)

    image_data = await image.read()
    image_buffer = io.BytesIO(image_data)

    image = Image.open(image_buffer)

    # save the image as png
    inputs = Segprocessor(images=image, return_tensors="pt").to(device)
    outputs = Segmodel(**inputs)
    logits = outputs.logits.cpu()

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0]

    result = {}
    for label, label_str in id2labels.items():
        cropped_image, cropped_image_with_white, score = extract_cropped_image_by_labels(
            image, pred_seg, [int(label)]
        )
        if cropped_image:
            if score > 0.01:
                print(f"Detected {label_str} with a score of {score}.")
                result[label_str] = score
                
    responses = []
    for label, _ in result.items():
        if label in ontology:
            responses.append(ontology[label])
    analysis = await generate_nutrient_analysis(responses)
    return JSONResponse(content={"nutrients": responses, "suggestions": analysis}, status_code=200)
  
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)