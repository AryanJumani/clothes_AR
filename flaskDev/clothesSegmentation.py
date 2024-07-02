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

    # Convert the mask to a binary numpy array (255 where clothing, 0 otherwise)
    binary_mask = clothing_mask.numpy().astype(np.uint8) * 255

    # Optionally invert the mask
    inverted_mask = np.where(binary_mask == 255, 0, 255).astype(np.uint8)  # Ensure uint8 type here

    # Convert the adjusted binary mask back to a PIL image for easier handling or visualization
    mask_image = Image.fromarray(inverted_mask)

    return mask_image

def inpaint_image(image, mask):
    if not isinstance(mask, np.ndarray):
        # If mask is a PIL image, convert it to a numpy array
        mask = np.array(mask)

        # Ensure the mask is in 8-bit format
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Convert original_image to BGR format for OpenCV if it's not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Apply inpainting
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)

    # Convert back to RGB for consistency with PIL/Image output
    inpainted_image = cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB)

    return inpainted_image


# Example usage
image_url = "https://m.media-amazon.com/images/I/61qTPaU7dYL._AC_SX679_.jpg"
from PIL import Image

# Load your original image and mask here
url = "https://m.media-amazon.com/images/I/61qTPaU7dYL._AC_SX679_.jpg"
response = requests.get(url, stream=True)
original_image = Image.open(response.raw)

mask = segment_image(original_image)
final_image = inpaint_image(original_image, mask)
cv2.imwrite('final_output.png', final_image)
cv2.imshow('Result', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

