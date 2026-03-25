from pydantic import BaseModel, Field
from typing import Optional


class Observation(BaseModel):
    email_id: str
    subject: str
    body: str
    sender: str
    task_id: int
    step_number: int


class Action(BaseModel):
    label: Optional[str] = Field(None, description="spam or not_spam")
    priority: Optional[str] = Field(None, description="urgent, normal, or low")
    category: Optional[str] = Field(None, description="billing, support, feedback, or other")
    reply: Optional[str] = Field(None, description="drafted reply text")


class RewardSignal(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    done: bool
    info: dict


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict
