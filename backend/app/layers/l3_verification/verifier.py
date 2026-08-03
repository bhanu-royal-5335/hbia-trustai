"""
Layer 3: Verifier
Orchestrates fact-checking, confidence scoring, and hallucination detection.
"""
import json
from dataclasses import dataclass, field
from typing import List, Optional
from openai import AsyncOpenAI
from app.core.config import settings
from app.layers.l1_retrieval.vector_store import RetrievedChunk
from app.layers.l2_generation.prompt_builder import prompt_builder
import structlog

logger = structlog.get_logger()


@dataclass
class ClaimResult:
    claim_text: str
    is_supported: bool
    confidence: float
    supporting_source: Optional[str] = None


@dataclass
class VerificationOutput:
    trust_score: float
    hallucination_risk: str  # low | medium | high
    claims: List[ClaimResult] = field(default_factory=list)
    verified_claims: int = 0
    total_claims: int = 0
    has_hallucinations: bool = False
    hallucinated_spans: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)


class Verifier:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def verify(
        self,
        response: str,
        retrieved_chunks: List[RetrievedChunk],
    ) -> VerificationOutput:
        """Full L3 verification pipeline."""

        if not retrieved_chunks:
            # No sources → medium risk baseline
            return VerificationOutput(
                trust_score=60.0,
                hallucination_risk="medium",
                total_claims=0,
                verified_claims=0,
            )

        # Step 1: Extract and verify claims
        claims = await self._extract_and_verify_claims(response, retrieved_chunks)

        # Step 2: Confidence scoring
        avg_retrieval = sum(c.relevance_score for c in retrieved_chunks) / len(retrieved_chunks)
        verified_count = sum(1 for c in claims if c.is_supported)
        total_claims = len(claims)
        claim_ratio = verified_count / total_claims if total_claims > 0 else 1.0

        # Weighted trust score
        retrieval_score = avg_retrieval * 30
        claim_score = claim_ratio * 50
        coherence_score = 20.0  # baseline
        trust_score = min(100.0, retrieval_score + claim_score + coherence_score)

        # Step 3: Hallucination risk
        if trust_score >= 80:
            risk = "low"
        elif trust_score >= 60:
            risk = "medium"
        else:
            risk = "high"

        unverified = [c.claim_text for c in claims if not c.is_supported]
        issues = [f"Unsupported claim: {c[:100]}" for c in unverified]

        return VerificationOutput(
            trust_score=round(trust_score, 1),
            hallucination_risk=risk,
            claims=claims,
            verified_claims=verified_count,
            total_claims=total_claims,
            has_hallucinations=len(unverified) > 0,
            hallucinated_spans=unverified,
            issues=issues,
        )

    async def _extract_and_verify_claims(
        self, response: str, chunks: List[RetrievedChunk]
    ) -> List[ClaimResult]:
        messages = prompt_builder.build_verification_prompt(response, chunks)
        try:
            api_response = await self.client.chat.completions.create(
                model=settings.verification_model,
                messages=messages,
                temperature=0.0,
                max_tokens=1500,
            )
            raw = api_response.choices[0].message.content.strip()
            # Clean up JSON
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            return [
                ClaimResult(
                    claim_text=item.get("claim", ""),
                    is_supported=item.get("is_supported", False),
                    confidence=float(item.get("confidence", 0.5)),
                    supporting_source=item.get("supporting_source"),
                )
                for item in data
            ]
        except Exception as e:
            logger.warning("claim_verification_failed", error=str(e))
            return []


verifier = Verifier()
