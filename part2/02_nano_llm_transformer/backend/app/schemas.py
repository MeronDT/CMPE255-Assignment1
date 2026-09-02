from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    temperature: float = Field(0.8, ge=0.1, le=2.0)
    top_p: float = Field(0.95, ge=0.1, le=1.0)
    max_new_tokens: int = Field(150, ge=1, le=384)


class ChatResponse(BaseModel):
    reply: str
    generation_ms: float
    tokens_generated: int
    tokens_per_sec: float
