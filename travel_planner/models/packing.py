from typing import List, Optional, Literal, Annotated
from pydantic import BaseModel, Field

class PackingTipsResponse(BaseModel):
    packing_list: Annotated[List[str], Field(description="List of items to pack (at least 6 items)")]
    tips: Annotated[List[str], Field(description="List of travel tips (at least 6 tips)")]
    recommendation: Annotated[Literal["GO", "DON'T GO"], Field(description="Final recommendation")]
    recommendation_reason: Annotated[str, Field(description="Detailed reason for the recommendation")]
    alternative: Annotated[Optional[str], Field(description="Alternative suggestion if DON'T GO, else null")]
