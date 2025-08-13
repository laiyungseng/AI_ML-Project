from pydantic import BaseModel
from typing import List


class PredictResponse(BaseModel):
    y_pred: List[float]
    EM: List[float]
  