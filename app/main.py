import sys

import uvicorn
from fastapi import FastAPI, File, UploadFile, Depends, Request
from src.models.predict_model import MODEL, predict_top_labels
from structures import OptionalK

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict(file: UploadFile = File(...), optional: OptionalK = Depends()):
    top_k = optional.dict()['k'] or 5
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = await file.read()
    prediction = predict_top_labels(model=MODEL, image_bytes=image, top_k=top_k)
    return prediction


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Run python server.py <HOST> <PORT>')
        sys.exit(1)

    host = sys.argv[1]
    port = int(sys.argv[2])

    uvicorn.run('server:app', host=host, port=port)