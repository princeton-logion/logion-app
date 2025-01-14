from pydantic import BaseModel, Field, field_validator
from typing import List, List

"""
Class for input
"""
class PredictionRequest(BaseModel):
    text: str
    model_name: str

    @field_validator("text")
    def check_mask_presence(cls, v):
        if "[MASK]" not in v:
            raise ValueError("Input must contain one or more [MASK] tokens.")
        return v
    
    @field_validator("text")
    def text_not_empty(cls, value):
        if not value.strip():
            raise ValueError("Input text cannot be empty.")
        return value

    @field_validator("model_name")
    def model_name_not_empty(cls, value):
        if not value.strip():
            raise ValueError("User must select model.")
        return value


"""
Block of classes for handling output
"""
class TokenPrediction(BaseModel):
    token: str
    probability: float

    @field_validator("token")
    def token_not_empty(cls, value):
        if not value.strip():
            raise ValueError("Predicted token cannot be an empty string value.")
        return value
    
    @field_validator("probability")
    def confidence_score_range(cls, value):
        if not 0 <= value <= 1:
             raise ValueError("Predicted probability score must be between 0 and 1.")
        return value

class MaskedIndexPredictions(BaseModel):
    predictions: List[TokenPrediction] = Field(..., min_items=1)

class PredictionResponse(BaseModel):
    predictions: dict[int, MaskedIndexPredictions]

    @field_validator('predictions')
    def validate_predictions(cls, predictions):
        if not isinstance(predictions, dict):
            raise ValueError("Mask predictions must be organized in a dictionary.")
        for masked_index, pred_list in predictions.items():
            if not isinstance(masked_index, int):
                raise ValueError("Mask index key must be an integer.")
            if not isinstance(pred_list, MaskedIndexPredictions):
                raise ValueError("Entry associated with mask index key must be a MaskedIndexPredictions list object.")
        return predictions

