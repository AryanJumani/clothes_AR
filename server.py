# image seg
import os
import shutil
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as nn
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from io import BytesIO

# db functionality
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response, JSONResponse
from sqlmodel import SQLModel, Field, Session, select, create_engine
import uvicorn
import bcrypt
import secret

db_pwd = secret.db_pwd

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = FastAPI()
engine = create_engine("mysql+pymysql://root:" + db_pwd + "@localhost/trion")

processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")


class User(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    email: str = Field(primary_key=True)
    pwd_hash: str


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

        # Return bytes
        buffer = BytesIO()
        result.save(buffer, format="PNG")
        return Response(content=buffer.getvalue(), media_type="image/png")
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/login")
def login(user: User):
    email = user.email
    pwd_hash = user.pwd_hash
    with Session(engine) as session:
        statement = select(User).where(User.email == email)
        user = session.exec(statement).first()
        if not user or not bcrypt.checkpw(pwd_hash.encode(), user.pwd_hash.encode()):
            return JSONResponse(status_code=401,
                                content={"status": "error", "message": "Invalid credentials"})
        return {"status": "success", "message": "Login successful"}


@app.post("/register")
def register(user: User):
    email = user.email
    pwd_hash = user.pwd_hash
    with Session(engine) as session:
        user = session.exec(select(User).where(User.email == email)).first()
        if user:
            return JSONResponse(
                status_code=409,
                content={"status": "error", "message": "User already exists"}
            )
        hashed = bcrypt.hashpw(pwd_hash.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        new = User(email=email, pwd_hash=hashed)
        session.add(new)
        session.commit()
        return {"status": "success", "message": "User registered successfully"}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
