"""
Layer 2: Prompt Builder
Constructs structured prompts for RAG generation, verification, and correction.
"""
from typing import List, Optional
from app.layers.l1_retrieval.vector_store import RetrievedChunk


SYSTEM_PROMPT = """You are HBIA-TrustAI, a trustworthy AI assistant. You answer questions accurately and honestly using the provided source documents.

CRITICAL RULES:
1. ONLY use information from the provided source documents. Do not invent facts.
2. If the sources don't contain enough information, say so clearly.
3. Always cite which source supports each key claim using [Source N] notation.
4. Be precise, clear, and factual.
5. If you are uncertain about something, explicitly state your uncertainty."""


class PromptBuilder:

    def build_rag_prompt(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
        chat_history: Optional[List[dict]] = None,
        use_chain_of_thought: bool = True,
    ) -> List[dict]:
        """Build messages list for RAG generation."""

        # Format sources
        sources_text = ""
        for i, chunk in enumerate(retrieved_chunks, 1):
            sources_text += f"\n[Source {i}] ({chunk.filename}, relevance: {chunk.relevance_score:.2f}):\n{chunk.content}\n"

        cot_instruction = ""
        if use_chain_of_thought:
            cot_instruction = "\n\nThink step by step before answering. First reason about what the sources say, then compose your answer."

        user_content = f"""<retrieved_sources>
{sources_text if sources_text else "No relevant sources found in the knowledge base."}
</retrieved_sources>

<user_query>
{query}
</user_query>
{cot_instruction}

Provide a comprehensive, accurate answer based solely on the retrieved sources. Cite sources using [Source N] notation."""

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if chat_history:
            for msg in chat_history[-6:]:  # Last 3 turns
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": user_content})
        return messages

    def build_verification_prompt(
        self,
        response: str,
        source_chunks: List[RetrievedChunk],
    ) -> List[dict]:
        """Build prompt for claim extraction and verification."""
        sources = "\n".join(
            f"[Source {i+1}]: {c.content[:500]}" for i, c in enumerate(source_chunks)
        )

        content = f"""Extract and verify each factual claim in this AI response.

<ai_response>
{response}
</ai_response>

<sources>
{sources}
</sources>

For each claim, check if it is supported by the sources.
Return JSON array:
[
  {{
    "claim": "<claim text>",
    "is_supported": true/false,
    "confidence": 0.0-1.0,
    "supporting_source": "<Source N or null>"
  }}
]
Return ONLY the JSON array, nothing else."""

        return [
            {"role": "system", "content": "You are a fact-checking assistant. Extract and verify claims."},
            {"role": "user", "content": content},
        ]

    def build_correction_prompt(
        self,
        original_query: str,
        original_response: str,
        issues: List[str],
        additional_chunks: List[RetrievedChunk],
    ) -> List[dict]:
        """Build prompt for self-correction."""
        issues_text = "\n".join(f"- {issue}" for issue in issues)
        additional_context = "\n".join(
            f"[Additional Source {i+1}]: {c.content}" for i, c in enumerate(additional_chunks)
        )

        content = f"""Your previous response had issues that need correction.

<original_query>{original_query}</original_query>

<previous_response>{original_response}</previous_response>

<identified_issues>
{issues_text}
</identified_issues>

<additional_context>
{additional_context if additional_context else "No additional sources available."}
</additional_context>

Please provide a corrected, improved response that:
1. Fixes all identified issues
2. Only states what is supported by sources
3. Clearly acknowledges any limitations
4. Uses [Source N] citations"""

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]


prompt_builder = PromptBuilder()
