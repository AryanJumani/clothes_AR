import cv2
import requests
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import torch.nn.functional as nn

def download_image(image_url):
    response = requests.get(image_url)
    if response.status_code != 200:
        return None
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image

def segment_image(image):
    # Assuming the model is loaded and ready, this function should return the mask of the hoodie
    # This is a placeholder for the actual segmentation logic
    processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

    # Load the image
    url = "https://m.media-amazon.com/images/I/61qTPaU7dYL._AC_SX679_.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # Process the image
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits.cpu()

    # Upsample the logits to match the image size
    upsampled_logits = nn.interpolate(
        logits,
        size=image.size[::-1],  # Size needs to be (height, width)
        mode="bilinear",
        align_corners=False,
    )

    # Get the segmentation mask
    pred_seg = upsampled_logits.argmax(dim=1)[0]
    clothing_mask = pred_seg == 4  # Assuming clothing is class 4

    image_np = np.array(image)

    # Create a white background
    background_white = np.ones_like(image_np) * 255

    # Copy the clothing regions from the original image to the new image with a white background
    for c in range(3):  # Assuming the image has 3 color channels (RGB)
        background_white[:, :, c] = np.where(clothing_mask, image_np[:, :, c], 255)

    # Convert the extracted clothing image back to PIL format
    extracted_clothing_image_white_bg = Image.fromarray(background_white)

    return extracted_clothing_image_white_bg

from PIL import Image

# Load your original image and mask here
url = "https://m.media-amazon.com/images/I/61qTPaU7dYL._AC_SX679_.jpg"
response = requests.get(url, stream=True)
original_image = Image.open(response.raw)

segmented_img = segment_image(original_image)
plt.imshow(segmented_img)
plt.axis("off")
plt.savefig("segmented_image.png", bbox_inches="tight")
# The segmented image will be saved as "segmented_image.png" in the current directory
