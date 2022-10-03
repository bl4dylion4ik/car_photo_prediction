import sys

import uvicorn
from fastapi import FastAPI, File, UploadFile, Request
from src.models.predict_model import MODEL, predict_top_labels

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        print("Image must be jpg or png format!")
        return "Image must be jpg or png format!"
    image = await file.read()
    prediction = predict_top_labels(model=MODEL, image_bytes=image, top_k=5)
    print(prediction)
    return prediction


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Run python server.py <HOST> <PORT>')
        sys.exit(1)

    host = sys.argv[1]
    port = int(sys.argv[2])

    uvicorn.run('server:app', host=host, port=port)