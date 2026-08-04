from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, List, Optional, Tuple, Annotated


"""
Classes for lacuna prediction task
"""


"""
Input Class
"""


class PredictionRequest(BaseModel):
    text: str
    model_name: str
    text_type: str = "prose"

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
    
    @field_validator("text_type")
    def validate_text_type(cls: type, value: str) -> str:
        allowed = {"prose", "hexameter"}
        if value not in allowed:
            raise ValueError("text_type must be one of the following: 'prose', 'hexameter'.")
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


class ScansionLine(BaseModel):
    """
    One display line of the hexameter scansion payload
    (predict_utils.restored_text_scansion), as consumed by the frontend
    ScansionDisplay component.  Named ScansionLine to avoid confusion
    with hex_filter.LineScansion, the scanner-internal NamedTuple this
    payload is derived from.

    Attributes:
        line (str) -- restored line text (every gap filled w/ its
            top-ranked prediction)
        syllables ( List[str] ) -- 1 display string per scanned syllable
        markers ( List[str] ) -- 1 scan marker (L/S/X) per syllable,
            index-aligned with syllables
        word_breaks ( List[int] ) -- syllable indices AFTER which a word
            boundary falls
        segments ( List[List[Tuple[str, bool]]] ) -- per syllable, its
            text split into (chars, is_prediction) runs; concatenation
            of a syllable's run chars == the syllable text (serializes
            to [[chars, bool], ...] arrays for the frontend)
        prediction_syllables ( List[int] ) -- syllables containing >= 1
            prediction character (coarse per-syllable flag)
    """
    line: str
    syllables: List[str]
    markers: List[str]
    word_breaks: List[int]
    segments: List[List[Tuple[str, bool]]]
    prediction_syllables: List[int]

    @model_validator(mode="after")
    def check_display_alignment(self) -> "ScansionLine":
        n = len(self.syllables)
        if len(self.markers) != n:
            raise ValueError("Scan markers must align 1:1 with syllables.")
        if len(self.segments) != n:
            raise ValueError("Character segments must align 1:1 with syllables.")
        for syllable, runs in zip(self.syllables, self.segments):
            if "".join(chars for chars, _ in runs) != syllable:
                raise ValueError(
                    "Segment character runs must concatenate to their syllable text."
                )
        for idx in self.word_breaks:
            if not 0 <= idx < n:
                raise ValueError("Word break index out of syllable range.")
        for idx in self.prediction_syllables:
            if not 0 <= idx < n:
                raise ValueError("Prediction syllable index out of syllable range.")
        allowed_markers = {"L", "S", "X"}
        for marker in self.markers:
            if marker not in allowed_markers:
                raise ValueError("Scan markers must be one of: 'L', 'S', 'X'.")
        return self


class PredictionResponse(BaseModel):
   predictions: Dict#[int, MaskedIndexPredictions]
   origText: Optional[str] = None
   # verse line scansion display payload (hexameter only)
   scansion: Optional[List[ScansionLine]] = None

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
                   "Value for mask index key must be a MaskedIndexPredictions list object."
               )
       return predictions