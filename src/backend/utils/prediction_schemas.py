from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Annotated


"""
Classes for lacuna prediction task
"""


"""
Input Class
"""


class PredictionRequest(BaseModel):
    text: str
    model_name: str

    @field_validator("text")
    def check_mask_presence(cls: type, value: str) -> str:
        if "-" not in value:
            raise ValueError("Input text must contain one or more '-' characters.")
        return value

    @field_validator("text")
    def text_not_null(cls: type, value: str) -> str:
        if not value.strip():
            raise ValueError("Input text cannot be an empty string.")
        return value

    @field_validator("model_name")
    def model_not_null(cls: type, value: str) -> str:
        if not value.strip():
            raise ValueError("No model selected. User must select model.")
        return value


"""
Output Classes
"""


class TokenPrediction(BaseModel):
    token: str
    probability: float

    #@field_validator("token")
    #def pred_token_not_null(cls: type, value: str) -> str:
        #if not value.strip():
            #raise ValueError("Predicted token cannot be an empty string.")
        #return value

    @field_validator("probability")
    def probability_score_range(cls: type, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("Predicted probability score must be between 0 and 1.")
        return value


class MaskedIndexPredictions(BaseModel):
    predictions: Annotated[List[TokenPrediction], Field(min_items=1)]


class PredictionResponse(BaseModel):
   predictions: Dict#[int, MaskedIndexPredictions]

   @field_validator("predictions")
   def validate_predictions(cls: type, predictions: Dict) -> Dict:
       if not isinstance(predictions, Dict):
           raise ValueError(
               "Mask token predictions must be organized as a dictionary."
           )
       for masked_index, pred_list in predictions.items():
           if not isinstance(masked_index, int):
               raise ValueError("Dictionary key for mask index must be an integer.")
           if not isinstance(pred_list, MaskedIndexPredictions):
               raise ValueError(
                   "Value for mask index key must be a MaskedIndexPredictions list"
                   " object."
               )
       return predictions
