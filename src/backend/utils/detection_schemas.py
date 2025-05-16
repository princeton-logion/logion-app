from typing import List, Annotated
from pydantic import BaseModel, field_validator, Field


"""
Classes for error detection task
"""


"""
Input Class
"""


class DetectionRequest(BaseModel):
    text: str
    model_name: str
    lev_distance: int

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

    @field_validator("lev_distance")
    def lev_dist_not_null(cls: type, value: int) -> int:
        if not value:
            raise ValueError(
                "No levenshtein distance selected. User must select levenshtein distance.")
        return value


"""
Output Classes
"""


class MaskPrediction(BaseModel):
    token: str
    probability: float

    @field_validator("token")
    def pred_token_not_null(cls: type, value: str) -> str:
        if not value.strip():
            raise ValueError("Predicted token cannot be an empty string value.")
        return value

    @field_validator("probability")
    def confidence_score_range(cls: type, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("Predicted confidence score must be between 0 and 1.")
        return value


class WordPrediction(BaseModel):
    original_word: str
    chance_score: float
    global_word_index: int
    suggestions: Annotated[List[MaskPrediction], Field(min_items=1)]

    @field_validator("original_word")
    def original_word_not_null(cls: type, value: str) -> str:
        if not value.strip():
            raise ValueError("Original masked word cannot be an empty string.")
        return value

    @field_validator("chance_score")
    def chance_score_range(cls: type, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("Predicted chance score must be between 0 and 1.")
        return value

    @field_validator("global_word_index")
    def global_word_index_type(cls: type, value: int) -> int:
        if not isinstance(value, int):
            raise ValueError("Global word index must be an integer.")
        return value


class CCRResult(BaseModel):
    ccr_value: float

    @field_validator("ccr_value")
    def ccr_score_type(cls: type, value: float) -> float:
        if not isinstance(value, (float, int)):
            raise ValueError("CCR score must be a float or integer.")
        return value


class DetectionResponse(BaseModel):
    predictions: List[WordPrediction]
    ccr: List[CCRResult]

    @field_validator("predictions")
    def predictions_not_null(cls: type, value: List) -> List:
        if not value:
            raise ValueError("Predictions list cannot be empty.")
        return value

    @field_validator("ccr")
    def ccr_not_null(cls: type, value: List) -> List:
        if not value:
            raise ValueError("CCR list cannot be empty.")
        return value
