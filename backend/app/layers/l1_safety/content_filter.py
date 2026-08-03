"""
Layer 1b: Content Safety Filter
Detects prompt injection, harmful content, and policy violations.
"""
import re
from dataclasses import dataclass, field
from typing import List
from openai import AsyncOpenAI
from app.core.config import settings
import structlog
import json

logger = structlog.get_logger()

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"forget\s+(everything|all)\s+above",
    r"you\s+are\s+now\s+(a|an)\s+(?!assistant)",
    r"act\s+as\s+if\s+you\s+(have\s+no|don't\s+have)",
    r"jailbreak",
    r"DAN\s+mode",
    r"system\s*:\s*you\s+are",
    r"<\s*system\s*>",
    r"\[INST\].*\[/INST\]",
]

HARMFUL_PATTERNS = [
    r"\b(bomb|explosive|weapon)\s+(making|building|creation|instructions)\b",
    r"\b(child|minors?)\s+(sexual|explicit|nude)\b",
    r"\bsynthes(is|ize)\s+.{0,30}(drug|chemical\s+weapon|poison)\b",
]


@dataclass
class SafetyResult:
    is_safe: bool
    violations: List[str] = field(default_factory=list)
    filtered_text: str = ""
    risk_level: str = "none"  # none | low | medium | high | critical


class ContentFilter:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def check_input(self, text: str) -> SafetyResult:
        violations = []
        risk_level = "none"

        # Regex checks
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append(f"Prompt injection attempt detected")
                risk_level = "critical"
                break

        for pattern in HARMFUL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append("Harmful content request detected")
                risk_level = "critical"
                break

        if violations:
            return SafetyResult(
                is_safe=False,
                violations=violations,
                filtered_text="",
                risk_level=risk_level,
            )

        # LLM-based check for subtle cases
        try:
            response = await self.client.moderations.create(input=text)
            result = response.results[0]
            if result.flagged:
                flagged_cats = [
                    cat for cat, flagged in result.categories.__dict__.items() if flagged
                ]
                return SafetyResult(
                    is_safe=False,
                    violations=[f"Content policy violation: {', '.join(flagged_cats)}"],
                    filtered_text="",
                    risk_level="high",
                )
        except Exception as e:
            logger.warning("moderation_api_failed", error=str(e))

        return SafetyResult(is_safe=True, filtered_text=text, risk_level="none")

    async def check_output(self, text: str) -> SafetyResult:
        try:
            response = await self.client.moderations.create(input=text)
            result = response.results[0]
            if result.flagged:
                return SafetyResult(
                    is_safe=False,
                    violations=["Output contains policy-violating content"],
                    filtered_text="",
                    risk_level="high",
                )
        except Exception as e:
            logger.warning("output_moderation_failed", error=str(e))

        return SafetyResult(is_safe=True, filtered_text=text, risk_level="none")


content_filter = ContentFilter()
