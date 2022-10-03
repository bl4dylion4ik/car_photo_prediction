from typing import List, Optional
from pydantic import BaseModel


class Prediction(BaseModel):
    labels_list: List[str]
    scores_list: List[float]


class ReturnValue(BaseModel):
    success: bool
    message: Optional[str]
    traceback: Optional[str]


class PredictReturnValue(ReturnValue):
    prediction: Optional[Prediction]