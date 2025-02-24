import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import requests
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import torch
import torch.nn.functional as nn

def download_image(image_url):
    response = requests.get(image_url)
    if response.status_code != 200:
        return None
    image = Image.open(requests.get(image_url, stream=True).raw)
    return image

processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

def segment_image(image):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits.cpu()
    upsampled_logits = nn.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    pred_seg = upsampled_logits.argmax(dim=1)[0]
    clothing_mask = pred_seg == 4
    image_np = np.array(image)
    background_white = np.zeros((image_np.shape[0], image_np.shape[1], 4), dtype=np.uint8)
    transparent_image[:, :, :3] = image_np[:, :, :3]
    transparent_image[:, :, 3] = np.where(clothing_mask, 255, 0)
    extracted_clothing_trans = Image.fromarray(transparent_image, mode="RGBA")
    return extracted_clothing_trans

url = "https://m.media-amazon.com/images/I/61wZCWANufL._AC_SY879_.jpg"
original_image = download_image(url)
#original_image = Image.open("person.jpg")
if original_image:
    segmented_img = segment_image(original_image)
    segmented_img.save("Resources/Shirts/extracted_image.jpg")
    print("Saved image as extracted_image.jpg in resources folder")
else:
    print("Failed to download image.")
