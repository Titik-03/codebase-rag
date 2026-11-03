import uuid
from typing import List

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """Identify a source file and its character range."""
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Store a question and its identifier."""
    question_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Store a reference answer and its supporting sources."""
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Validate a collection of reference or unanswered questions."""
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Store the ranked sources retrieved for a question."""
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Store retrieved sources and a generated answer."""
    answer: str


class StudentSearchResults(BaseModel):
    """Validate batch retrieval output and the requested result count."""
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Validate batch answer output and the requested result count."""
    search_results: List[MinimalAnswer]
    k: int
