"""
Layer 0: Intent Classifier
Classifies user queries into intent types to guide the orchestration pipeline.
Now accepts NLP analysis from the NLP Analyzer as soft prior signals.
"""
from enum import Enum
from dataclasses import dataclass
from typing import Optional
try:
    from openai import AsyncOpenAI
except Exception:
    AsyncOpenAI = None
from app.core.config import settings
import structlog
import json

logger = structlog.get_logger()


class IntentType(str, Enum):
    FACTUAL_QUERY = "factual_query"
    CREATIVE = "creative"
    CODE_GENERATION = "code_generation"
    CONVERSATIONAL = "conversational"
    DOCUMENT_ANALYSIS = "document_analysis"
    COMPARISON = "comparison"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWER = "question_answer"


@dataclass
class IntentClassification:
    intent_type: IntentType
    confidence: float
    retrieval_strategy: str  # semantic | hybrid | keyword | none
    quality_threshold: float
    chain_of_thought: bool
    reasoning: str


INTENT_PROMPTS = {
    IntentType.FACTUAL_QUERY: {"threshold": 80.0, "retrieval": "hybrid", "cot": True},
    IntentType.CREATIVE: {"threshold": 60.0, "retrieval": "none", "cot": False},
    IntentType.CODE_GENERATION: {"threshold": 70.0, "retrieval": "semantic", "cot": True},
    IntentType.CONVERSATIONAL: {"threshold": 55.0, "retrieval": "none", "cot": False},
    IntentType.DOCUMENT_ANALYSIS: {"threshold": 85.0, "retrieval": "semantic", "cot": True},
    IntentType.COMPARISON: {"threshold": 80.0, "retrieval": "hybrid", "cot": True},
    IntentType.SUMMARIZATION: {"threshold": 75.0, "retrieval": "semantic", "cot": False},
    IntentType.QUESTION_ANSWER: {"threshold": 80.0, "retrieval": "hybrid", "cot": True},
}


class IntentClassifier:
    def __init__(self):
        self.api_key = settings.openai_api_key
        self.client = AsyncOpenAI(api_key=self.api_key) if (AsyncOpenAI and self.api_key) else None

    # ── Intent hint → IntentType mapping ─────────────────────────────────────
    _HINT_TO_INTENT = {
        "code_generation": IntentType.CODE_GENERATION,
        "summarization": IntentType.SUMMARIZATION,
        "comparison": IntentType.COMPARISON,
        "creative": IntentType.CREATIVE,
        "conversational": IntentType.CONVERSATIONAL,
        "factual_query": IntentType.FACTUAL_QUERY,
        "question_answer": IntentType.QUESTION_ANSWER,
        "document_analysis": IntentType.DOCUMENT_ANALYSIS,
    }

    def _resolve_from_nlp(self, nlp_analysis) -> Optional[IntentClassification]:
        """
        If NLP hint signals are very confident and unambiguous, return a
        pre-computed IntentClassification without an LLM call.
        Returns None if signals are ambiguous or absent.
        """
        if nlp_analysis is None:
            return None

        hints = getattr(nlp_analysis, "intent_hints", [])
        if not hints:
            return None

        # Only fast-path when there is exactly one clear signal
        if len(hints) == 1:
            intent = self._HINT_TO_INTENT.get(hints[0])
            if intent:
                config = INTENT_PROMPTS.get(intent, INTENT_PROMPTS[IntentType.QUESTION_ANSWER])
                logger.info("intent_resolved_from_nlp", intent=intent, hint=hints[0])
                return IntentClassification(
                    intent_type=intent,
                    confidence=0.82,
                    retrieval_strategy=config["retrieval"],
                    quality_threshold=config["threshold"],
                    chain_of_thought=config["cot"],
                    reasoning=f"NLP hint: {hints[0]}",
                )
        return None

    async def classify(self, query: str, nlp_analysis=None) -> IntentClassification:
        """Classify user query intent, optionally using NLP pre-analysis."""

        # Fast-path: resolve intent directly from unambiguous NLP signals
        nlp_result = self._resolve_from_nlp(nlp_analysis)
        if nlp_result and not self.client:
            # No LLM available — use NLP result directly
            return nlp_result

        # Build context-enriched prompt
        nlp_context_str = ""
        if nlp_analysis:
            hints = getattr(nlp_analysis, "intent_hints", [])
            topics = getattr(nlp_analysis, "topics", [])
            entities = [e.text for e in getattr(nlp_analysis, "entities", [])[:5]]
            sentiment = getattr(nlp_analysis, "sentiment", "neutral")
            complexity = getattr(nlp_analysis, "complexity", "moderate")
            nlp_context_str = (
                f"\n\nPre-analysis NLP signals (use these as soft prior hints):"
                f"\n- Detected topics: {topics or 'none'}"
                f"\n- Detected entities: {entities or 'none'}"
                f"\n- Sentiment: {sentiment}"
                f"\n- Complexity: {complexity}"
                f"\n- Intent hints: {hints or 'none'}"
            )

        prompt = f"""Analyze this user query and classify its intent. Return a JSON object.

Query: "{query}"

Classify into one of these intents:
- factual_query: Looking for specific facts or information
- creative: Creative writing, brainstorming, generating content
- code_generation: Writing, debugging, or explaining code
- conversational: Small talk, greetings, casual questions
- document_analysis: Analyzing or extracting info from documents
- comparison: Comparing options, pros/cons analysis
- summarization: Requesting a summary of content
- question_answer: General question needing an informative answer
{nlp_context_str}

Return ONLY valid JSON:
{{
  "intent_type": "<one of the above>",
  "confidence": <0.0-1.0>,
  "reasoning": "<brief explanation>"
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=settings.verification_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            intent = IntentType(data.get("intent_type", "question_answer"))
            config = INTENT_PROMPTS.get(intent, INTENT_PROMPTS[IntentType.QUESTION_ANSWER])

            raw_confidence = float(data.get("confidence", 0.8))

            # Boost confidence if LLM agrees with NLP hint
            if nlp_result and nlp_result.intent_type == intent:
                raw_confidence = min(raw_confidence + 0.08, 1.0)

            return IntentClassification(
                intent_type=intent,
                confidence=raw_confidence,
                retrieval_strategy=config["retrieval"],
                quality_threshold=config["threshold"],
                chain_of_thought=config["cot"],
                reasoning=data.get("reasoning", ""),
            )
        except Exception as e:
            logger.warning("intent_classification_failed", error=str(e))
            # If NLP gave us a result, use it as fallback before hardcoded defaults
            if nlp_result:
                return nlp_result
            return IntentClassification(
                intent_type=IntentType.QUESTION_ANSWER,
                confidence=0.7,
                retrieval_strategy="hybrid",
                quality_threshold=settings.min_trust_score,
                chain_of_thought=True,
                reasoning="Fallback classification",
            )
