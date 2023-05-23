from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import os
import pandas as pd

import os
import torch
from PIL import Image
import matplotlib.pyplot as plt

import cv2
import numpy as np
import io

app = FastAPI(title='Deploying a banana model')

@app.get('/')
def read_root():
    return {"welcome_message": "welcome to my API!"}

@app.post("/predict")
def prediction(file: UploadFile = File(...)):
    # validate input file
    filename = file.filename
    fileExtension = filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not fileExtension:
        raise HTTPException(status_code=415, detail="Unsupported file provided.")
    # 
       # Read image as a stream of bytes
    image_stream = io.BytesIO(file.file.read())
    
    # Start the stream from the beginning (position zero)
    image_stream.seek(0)
    
    # Write the stream of bytes into a numpy array
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    
    # Decode the numpy array as an image
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    print(type(image))
    print(image.shape)
    cv2.imwrite('a.jpg', image)
    # load model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='Final_model/best_params/best.pt') 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image)
    img2 = results.render()
    # cv2.imwrite('asdf.jpg', cv2.cvtColor(img2[0], cv2.COLOR_RGB2BGR)) # If color scheme of loaded image is RGB
    cv2.imwrite('asdf.jpg', cv2.cvtColor(img2[0], cv2.COLOR_RGB2BGR))
    print("Done !!!")
    file_image = open('asdf.jpg', mode="rb")
    return StreamingResponse(file_image, media_type="image/jpeg")

