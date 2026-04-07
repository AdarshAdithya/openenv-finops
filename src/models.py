from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class Resource(BaseModel):
    id: str
    type: str
    cost: float
    utilization: float
    is_prod: bool = False

class Observation(BaseModel):
    resources: List[Resource]
    total_cost: float

class Action(BaseModel):
    cmd: Literal["terminate", "resize", "nop"]
    target_id: Optional[str] = None