from typing import List, Optional
from pydantic import BaseModel


class TOPK(BaseModel):
    k: Optional[int]
    desc: Optional[str] = 'Use number of top k probabilities'


class ReturnValue(BaseModel):
    success: bool
    message: Optional[str]
    traceback: Optional[str]


class PredictReturnValue(ReturnValue):
    prediction: Optional[dict]
