import os
import shutil
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as nn
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import uvicorn

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = FastAPI()

processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")


@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    try:
        # Save uploaded image temporarily
        temp_path = "temp_input.png"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        image = Image.open(temp_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits.cpu()
        upsampled_logits = nn.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        pred_seg = upsampled_logits.argmax(dim=1)[0]

        mask = (pred_seg == 4) | (pred_seg == 7)
        image_np = np.array(image)
        rgba_image = np.zeros((image_np.shape[0], image_np.shape[1], 4), dtype=np.uint8)
        rgba_image[:, :, :3] = image_np
        rgba_image[:, :, 3] = np.where(mask, 255, 0)

        result = Image.fromarray(rgba_image, mode="RGBA")
        bbox = result.getbbox()
        if bbox:
            result = result.crop(bbox)

        # Save final output to assets/web/shirt.png
        output_path = "assets/web/shirt.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result.save(output_path)

        return JSONResponse({"status": "success", "path": output_path})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


if __name__ == "main":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
