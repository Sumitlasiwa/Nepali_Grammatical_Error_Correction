from pydantic import BaseModel
from typing import List, Optional


class GrammarRequest(BaseModel):
    text: str
    model: str = "mt5"    # default to mt5; accepts "mt5" or "mbart"


class GrammarResponse(BaseModel):
    original: str
    corrected: str


class CandidateScore(BaseModel):
    rank: int
    sentence: str
    wins: int                        # tournament reranker gives wins, not scores
    final_score: Optional[float] = None   # kept optional for backwards compat
    mt5_score:   Optional[float] = None
    bert_score:  Optional[float] = None


class SentenceResult(BaseModel):
    input: str
    best_output: str
    all_candidates: List[CandidateScore]


class GrammarResponseDetailed(BaseModel):
    results: List[SentenceResult]

