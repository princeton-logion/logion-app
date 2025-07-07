from typing import Dict, Any, Type, Literal, Optional
from pydantic import BaseModel, field_validator
import uuid 

"""
Classes for WebSockets message schemas
"""

class BaseMsg(BaseModel):
    type: str
    task_id: str

    @field_validator("type")
    def type_allowed(cls: Type['BaseMsg'], value: str) -> str:
        MSG_TYPES = {"error", "ack", "progress", "cancelled", "result"}
        if value not in MSG_TYPES:
            raise ValueError(f"Message type not recognized: {value}")
        return value

    @field_validator("task_id")
    def task_id_not_null(cls: Type['BaseMsg'], value: str) -> str:
        if not value.strip():
            raise ValueError("Task ID cannot be null.")
        return value

    @field_validator("task_id")
    def task_id_valid_uuid(cls: Type['BaseMsg'], value: str) -> str:
        try:
            uuid.UUID(value)
        except ValueError:
            raise ValueError(f"Task ID not a valid UUID: {value}")
        return value

"""
User requests
"""
class UserTaskRequest(BaseMsg):
    request_data: Dict[str, Any]

    @field_validator("request_data")
    def data_not_null(cls: Type['UserTaskRequest'], value: Dict[str, Any]) -> Dict[str, Any]:
        if not value:
            raise ValueError("Task request data cannot be null.")
        return value

class UserCancelRequest(BaseMsg):
    pass

"""
Server messages
"""

class ServerProgressMsg(BaseMsg):
    type: Literal["progress"] = "progress"
    percentage: float
    message: str

    @field_validator("percentage")
    def percentage_range(cls: Type['ServerProgressMsg'], value: float) -> float:
        if not 0.0 <= value <= 100.0:
            raise ValueError(f"Invalid progress percentage: 0 < {round(value, 1)} < 100.")
        return value

    @field_validator("message")
    def message_not_null(cls: Type['ServerProgressMsg'], value: str) -> str:
        if not value.strip():
            raise ValueError("Progress message cannot be null.")
        return value

class ServerResultMsg(BaseMsg):
    type: Literal["result"] = "result"
    data: Dict[str, Any]

    @field_validator("data")
    def data_not_null(cls: Type['ServerResultMsg'], value: Dict[str, Any]) -> Dict[str, Any]:
        if not value:
            raise ValueError("Task result data cannot be null.")
        return value

class ServerErrorMsg(BaseMsg):
    type: Literal["error"] = "error"
    detail: str
    status_code: Optional[int] = None

    @field_validator("detail")
    def detail_not_null(cls: Type['ServerErrorMsg'], value: str) -> str:
        if not value.strip():
            raise ValueError("Error message cannot be null.")
        return value

class ServerAckMsg(BaseMsg):
    type: Literal["ack"] = "ack"
    message: str

    @field_validator("message")
    def message_not_null(cls: Type['ServerAckMsg'], value: str) -> str:
        if not value.strip():
            raise ValueError("Ack message cannot be null.")
        return value

class ServerCancelMsg(BaseMsg):
    type: Literal["cancelled"] = "cancelled"
    pass