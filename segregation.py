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
import mediapipe as mp


def download_image(image_url):
    response = requests.get(image_url)
    if response.status_code != 200:
        return None
    image = Image.open(requests.get(image_url, stream=True).raw)
    extracted_clothing_trans = image
    target_width = 440
    w, h = extracted_clothing_trans.size
    aspect_ratio = h / w
    new_height = int(target_width * aspect_ratio)
    resized_img = extracted_clothing_trans.resize((target_width, new_height), Image.LANCZOS)
    resized_img.save("Resources/Shirts/unextract.jpg")

    return resized_img

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
    clothing_mask = (pred_seg == 4) | (pred_seg == 7)
    image_np = np.array(image)
    transparent_image = np.zeros((image_np.shape[0], image_np.shape[1], 4), dtype=np.uint8)
    transparent_image[:, :, :3] = image_np[:, :, :3]
    transparent_image[:, :, 3] = np.where(clothing_mask, 255, 0)
    extracted_clothing_trans = Image.fromarray(transparent_image, mode="RGBA")

    bbox = extracted_clothing_trans.getbbox()
    
    if bbox:
        extracted_clothing_trans = extracted_clothing_trans.crop(bbox)

    target_width = 440
    w, h = extracted_clothing_trans.size
    aspect_ratio = h / w
    new_height = int(target_width * aspect_ratio)
    resized_img = extracted_clothing_trans.resize((target_width, new_height), Image.LANCZOS)

    return resized_img

url = "https://cdn.shopify.com/s/files/1/0098/8822/files/TrainingOversizedFleeceHoodieGSCalmPinkB5A7N-KCPD6678_3840x.jpg"
original_image = download_image(url)
#original_image = Image.open("person.jpg")
if original_image:
    segmented_img = segment_image(original_image)
    segmented_img.save("Resources/Shirts/extracted_image.png")
    print("Saved image as extracted_image.png in resources folder")
else:
    print("Failed to download image.")
