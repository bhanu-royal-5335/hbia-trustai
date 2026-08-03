"""
Layer 4: Self-Correction Engine
Identifies gaps and refines responses when trust score is below threshold.
"""
from dataclasses import dataclass, field
from typing import List, Optional
from app.layers.l1_retrieval.vector_store import RetrievedChunk
from app.layers.l3_verification.verifier import VerificationOutput
from app.layers.l2_generation.llm_client import llm_client, GenerationResult
from app.layers.l2_generation.prompt_builder import prompt_builder
from app.core.config import settings
import structlog

logger = structlog.get_logger()


@dataclass
class GapAnalysis:
    needs_correction: bool
    gaps: List[str] = field(default_factory=list)
    correction_instructions: List[str] = field(default_factory=list)
    refined_query: Optional[str] = None


class RefinementEngine:
    async def analyze_gaps(
        self,
        query: str,
        verification: VerificationOutput,
    ) -> GapAnalysis:
        if verification.trust_score >= settings.min_trust_score:
            return GapAnalysis(needs_correction=False)

        gaps = []
        instructions = []

        if verification.has_hallucinations:
            gaps.extend(verification.hallucinated_spans[:3])
            instructions.append(
                "Remove or correct claims that are not supported by source documents"
            )

        if verification.total_claims > 0:
            unsupported_ratio = 1 - (verification.verified_claims / verification.total_claims)
            if unsupported_ratio > 0.3:
                instructions.append(
                    "Reduce unsupported claims and add explicit uncertainty statements"
                )

        if verification.trust_score < 60:
            instructions.append(
                "Significantly revise the response to only include clearly supported facts"
            )
            gaps.append("Response may not be sufficiently grounded in retrieved sources")

        refined_query = f"{query} (provide only factually supported information)"

        return GapAnalysis(
            needs_correction=True,
            gaps=gaps,
            correction_instructions=instructions,
            refined_query=refined_query,
        )

    async def refine(
        self,
        original_query: str,
        original_response: str,
        gap_analysis: GapAnalysis,
        additional_chunks: List[RetrievedChunk],
        iteration: int,
    ) -> GenerationResult:
        logger.info(
            "self_correction_iteration",
            iteration=iteration,
            gaps=len(gap_analysis.gaps),
        )

        issues = gap_analysis.gaps + gap_analysis.correction_instructions
        messages = prompt_builder.build_correction_prompt(
            original_query,
            original_response,
            issues,
            additional_chunks,
        )

        result = await llm_client.generate(
            messages=messages,
            temperature=0.2,
            max_tokens=2000,
        )
        return result


refinement_engine = RefinementEngine()
