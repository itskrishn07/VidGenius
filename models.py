from typing import List, Optional, Any
from pydantic import BaseModel, Field


class ActionItem(BaseModel):
    task: str = Field(description="Description of the action item/task")
    owner: str = Field(default="Not specified", description="Person responsible for the task")
    deadline: str = Field(default="Not specified", description="Deadline or target completion date")


class MeetingInsights(BaseModel):
    action_items: str = Field(description="Formatted action items from the meeting")
    key_decisions: str = Field(description="Formatted list of key decisions")
    open_questions: str = Field(description="Formatted list of unresolved questions")


class PipelineResult(BaseModel):
    title: str = Field(description="Auto-generated title of the meeting/video")
    summary: str = Field(description="Synthesized bullet-point summary")
    transcript: str = Field(description="Complete transcript text")
    action_items: str = Field(description="Extracted action items")
    key_decisions: str = Field(description="Extracted key decisions")
    open_questions: str = Field(description="Extracted open questions")
    rag_chain: Optional[Any] = Field(default=None, description="LangChain RAG Runnable chain")

    class Config:
        arbitrary_types_allowed = True
