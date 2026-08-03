"""
Layer 0: Intent Classifier
Classifies user queries into intent types to guide the orchestration pipeline.
"""
from enum import Enum
from dataclasses import dataclass
from openai import AsyncOpenAI
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
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def classify(self, query: str) -> IntentClassification:
        """Classify user query intent using LLM."""
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

            return IntentClassification(
                intent_type=intent,
                confidence=float(data.get("confidence", 0.8)),
                retrieval_strategy=config["retrieval"],
                quality_threshold=config["threshold"],
                chain_of_thought=config["cot"],
                reasoning=data.get("reasoning", ""),
            )
        except Exception as e:
            logger.warning("intent_classification_failed", error=str(e))
            return IntentClassification(
                intent_type=IntentType.QUESTION_ANSWER,
                confidence=0.7,
                retrieval_strategy="hybrid",
                quality_threshold=settings.min_trust_score,
                chain_of_thought=True,
                reasoning="Fallback classification",
            )
