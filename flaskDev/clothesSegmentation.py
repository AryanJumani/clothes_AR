import requests as requests
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import requests
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

# Load the processor and model
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

# Load the image
url = "https://m.media-amazon.com/images/I/61qTPaU7dYL._AC_SX679_.jpg"
image = Image.open(requests.get(url, stream=True).raw)


# Process the image
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits.cpu()

# Upsample the logits
upsampled_logits = nn.functional.interpolate(
    logits,
    size=image.size[::-1],
    mode="bilinear",
    align_corners=False,
)

# Get the segmentation mask
pred_seg = upsampled_logits.argmax(dim=1)[0]
clothing_mask = pred_seg == 4

# Convert the mask to a numpy array and resize to match the image dimensions
clothing_mask = clothing_mask.numpy().astype(np.uint8)

# Convert the original image to a numpy array
image_np = np.array(image)

# Create a white background
background_white = np.ones_like(image_np) * 255

# Copy the clothing regions from the original image to the new image with a white background
for c in range(3):  # Assuming the image has 3 color channels (RGB)
    background_white[:, :, c] = np.where(clothing_mask, image_np[:, :, c], 255)

# Convert the extracted clothing image back to PIL format
extracted_clothing_image_white_bg = Image.fromarray(background_white)

# Display the extracted clothing image with a white background
plt.imshow(extracted_clothing_image_white_bg)
plt.axis('off')
plt.show()
