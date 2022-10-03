import sys
import traceback

import uvicorn
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException

from src.models.predict_model import MODEL, predict_top_labels
from structures import TOPK, ReturnValue, PredictReturnValue

app = FastAPI(title='Prediction popular car image')


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict", response_model=PredictReturnValue, name='Predict')
async def predict(file: UploadFile = File(...), options: TOPK = Depends()):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        raise HTTPException(status_code=400, detail=f'File \'{file.filename}\' is not an image.')

    try:
        top_k = options.dict()['k'] or 5
        image = await file.read()

        prediction = predict_top_labels(model=MODEL, image_bytes=image, top_k=top_k)
        return PredictReturnValue(success=True,
                                  prediction=prediction['prediction'])

    except Exception as error:
        return PredictReturnValue(
            success=False,
            message=str(error),
            traceback=str(traceback.format_exc()),
        )


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Run python server.py <HOST> <PORT>')
        sys.exit(1)

    host = sys.argv[1]
    port = int(sys.argv[2])

    uvicorn.run('server:app', host=host, port=port)
