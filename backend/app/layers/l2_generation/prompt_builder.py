"""
Layer 2: Prompt Builder
Constructs structured prompts for RAG generation, verification, and correction.
Now injects NLP analysis context to help the LLM produce more targeted responses.
"""
from typing import Dict, List, Optional
from app.layers.l1_retrieval.vector_store import RetrievedChunk


SYSTEM_PROMPT = """You are HBIA-TrustAI, an advanced, highly intelligent, and trustworthy AI assistant similar to ChatGPT and Claude. 

Your goal is to provide accurate, comprehensive, beautifully structured, and helpful responses in clear Markdown format.

GUIDELINES:
1. Use clear Markdown structure: headers (##, ###), bold text for emphasis, bullet points, and code blocks where relevant.
2. Ground your response in the provided retrieved sources (documents & live web results). Cite sources using [Source N] or [Domain] notation.
3. Provide direct, informative, and well-organized answers. For biography, historical, or explanatory queries, provide rich context, key facts, and clear takeaways.
4. If sources are limited, combine grounded factual statements with general knowledge, explicitly clarifying any uncertainties.
5. Maintain a professional, articulate, and engaging tone at all times."""


class PromptBuilder:

    def build_rag_prompt(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
        chat_history: Optional[List[dict]] = None,
        use_chain_of_thought: bool = True,
        nlp_context: Optional[Dict] = None,
    ) -> List[dict]:
        """Build messages list for RAG generation, optionally with NLP context."""

        # Format sources
        sources_text = ""
        for i, chunk in enumerate(retrieved_chunks, 1):
            sources_text += f"\n[Source {i}] ({chunk.filename}, relevance: {chunk.relevance_score:.2f}):\n{chunk.content}\n"

        cot_instruction = ""
        if use_chain_of_thought:
            cot_instruction = "\n\nThink step by step before answering. First reason about what the sources say, then compose your answer."

        # Build NLP context block
        nlp_block = ""
        if nlp_context:
            entities = nlp_context.get("entities", [])
            keywords = nlp_context.get("keywords", [])
            topics = nlp_context.get("topics", [])
            sentiment = nlp_context.get("sentiment", "neutral")
            question_type = nlp_context.get("question_type")
            complexity = nlp_context.get("complexity", "moderate")
            word_count = nlp_context.get("word_count", 0)

            entity_str = ", ".join(
                f"{e['text']} ({e['type']})" for e in entities[:6]
            ) if entities else "none detected"

            kw_str = ", ".join(keywords[:8]) if keywords else "none"
            topic_str = ", ".join(topics) if topics else "general"

            nlp_block = f"""\n<nlp_analysis>
Query metadata (auto-detected, use to tailor your response):
- Detected entities: {entity_str}
- Key topics: {topic_str}
- Top keywords: {kw_str}
- Sentiment: {sentiment}
- Question type: {question_type or 'statement/request'}
- Query complexity: {complexity} ({word_count} words)
</nlp_analysis>\n"""

        user_content = f"""{nlp_block}<retrieved_sources>
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
