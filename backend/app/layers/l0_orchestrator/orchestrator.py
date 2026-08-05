"""
Layer 0: HBIA Orchestrator
The central controller that coordinates all HBIA layers for trustworthy AI responses.

Pipeline:
  L0a: NLP Analysis — tokenise, extract entities/keywords/sentiment/topics
  L0b: Classify intent → route config (NLP-informed)
  L1a: Safety check input
  L1b: Hybrid RAG retrieval
  L2: LLM generation (NLP-context injected into prompt)
  L3: Verification + trust scoring
  L4: Self-correction loop (if below threshold)
  → Return response with full audit data + NLP insights
"""
import time
from dataclasses import dataclass, field
from typing import List, Optional, AsyncGenerator
import structlog

from app.core.config import settings
from app.layers.l0_orchestrator.intent_classifier import IntentClassifier, IntentClassification
from app.layers.l0_orchestrator.nlp_analyzer import nlp_analyzer, NLPAnalysis
from app.layers.l1_retrieval.retriever import rag_retriever
from app.layers.l1_retrieval.vector_store import RetrievedChunk
from app.layers.l1_safety.content_filter import content_filter
from app.layers.l2_generation.llm_client import llm_client, GenerationResult
from app.layers.l2_generation.prompt_builder import prompt_builder
from app.layers.l3_verification.verifier import verifier, VerificationOutput
from app.layers.l4_correction.refinement_engine import refinement_engine
from app.schemas.chat import SourceCitation

logger = structlog.get_logger()


@dataclass
class HBIAResponse:
    response: str
    trust_score: float
    verification: VerificationOutput
    sources: List[SourceCitation]
    correction_iterations: int
    latency_ms: int
    model_used: str
    token_usage: dict
    intent: Optional[str] = None
    layer_traces: dict = field(default_factory=dict)
    nlp_analysis: Optional[dict] = None


class HBIAOrchestrator:
    def __init__(self):
        self.intent_classifier = IntentClassifier()

    async def process(
        self,
        query: str,
        chat_history: Optional[List[dict]] = None,
        use_web_search: bool = True,
    ) -> HBIAResponse:
        """Full HBIA pipeline processing."""
        start_time = time.time()
        layer_traces = {}
        correction_iterations = 0
        chat_history = chat_history or []

        # ── L0a: NLP Analysis ─────────────────────────────────────────────
        t_nlp = time.time()
        nlp = nlp_analyzer.analyze(query)
        layer_traces["l0_nlp"] = {
            **nlp.to_dict(),
            "latency_ms": int((time.time() - t_nlp) * 1000),
        }
        logger.info(
            "l0_nlp_complete",
            entities=len(nlp.entities),
            keywords=nlp.keywords[:5],
            sentiment=nlp.sentiment,
            topics=nlp.topics,
            complexity=nlp.complexity,
        )

        # ── L0b: Intent Classification (NLP-informed) ─────────────────────
        t0 = time.time()
        intent = await self.intent_classifier.classify(query, nlp_analysis=nlp)
        layer_traces["l0_intent"] = {
            "intent": intent.intent_type,
            "confidence": intent.confidence,
            "strategy": intent.retrieval_strategy,
            "threshold": intent.quality_threshold,
            "latency_ms": int((time.time() - t0) * 1000),
        }
        logger.info("l0_complete", intent=intent.intent_type, confidence=intent.confidence)

        # ── L1b: Safety Check ────────────────────────────────────────────
        t1 = time.time()
        safety = await content_filter.check_input(query)
        layer_traces["l1_safety"] = {
            "is_safe": safety.is_safe,
            "risk_level": safety.risk_level,
            "latency_ms": int((time.time() - t1) * 1000),
        }
        if not safety.is_safe:
            return self._blocked_response(
                reason=", ".join(safety.violations),
                start_time=start_time,
                layer_traces=layer_traces,
            )

        # ── L1a: RAG Retrieval (Vector + Web Search) ───────────────────────
        t2 = time.time()
        retrieved_chunks: List[RetrievedChunk] = []
        if intent.retrieval_strategy != "none":
            retrieved_chunks = await rag_retriever.retrieve(
                query=query,
                k=settings.max_retrieval_docs,
                strategy=intent.retrieval_strategy,
                use_web_search=use_web_search,
            )
        layer_traces["l1_retrieval"] = {
            "chunks_retrieved": len(retrieved_chunks),
            "strategy": intent.retrieval_strategy,
            "use_web_search": use_web_search,
            "avg_relevance": (
                sum(c.relevance_score for c in retrieved_chunks) / len(retrieved_chunks)
                if retrieved_chunks else 0
            ),
            "latency_ms": int((time.time() - t2) * 1000),
        }
        logger.info("l1_retrieval_complete", chunks=len(retrieved_chunks))

        # ── L2: Generation (with NLP context) ───────────────────────────
        t3 = time.time()
        messages = prompt_builder.build_rag_prompt(
            query=query,
            retrieved_chunks=retrieved_chunks,
            chat_history=chat_history,
            use_chain_of_thought=intent.chain_of_thought,
            nlp_context=nlp.to_dict(),
        )
        gen_result: GenerationResult = await llm_client.generate(
            messages=messages,
            temperature=0.3,
            max_tokens=2000,
        )
        current_response = gen_result.content
        layer_traces["l2_generation"] = {
            "model": gen_result.model_used,
            "tokens": gen_result.prompt_tokens + gen_result.completion_tokens,
            "latency_ms": int((time.time() - t3) * 1000),
        }
        logger.info("l2_generation_complete", model=gen_result.model_used)

        # ── L3: Verification ─────────────────────────────────────────────
        t4 = time.time()
        verification = await verifier.verify(current_response, retrieved_chunks)
        layer_traces["l3_verification"] = {
            "trust_score": verification.trust_score,
            "risk": verification.hallucination_risk,
            "claims": verification.total_claims,
            "verified": verification.verified_claims,
            "latency_ms": int((time.time() - t4) * 1000),
        }
        logger.info(
            "l3_verification_complete",
            trust_score=verification.trust_score,
            risk=verification.hallucination_risk,
        )

        # ── L4: Self-Correction Loop ──────────────────────────────────────
        while (
            verification.trust_score < intent.quality_threshold
            and correction_iterations < settings.max_correction_iterations
        ):
            correction_iterations += 1
            t5 = time.time()
            logger.info(
                "l4_correction_starting",
                iteration=correction_iterations,
                trust_score=verification.trust_score,
            )

            gap_analysis = await refinement_engine.analyze_gaps(query, verification)
            if not gap_analysis.needs_correction:
                break

            # Re-retrieve with refined query
            additional_chunks = await rag_retriever.retrieve(
                query=gap_analysis.refined_query or query,
                k=settings.max_retrieval_docs,
                strategy="hybrid",
            )

            refined = await refinement_engine.refine(
                original_query=query,
                original_response=current_response,
                gap_analysis=gap_analysis,
                additional_chunks=additional_chunks,
                iteration=correction_iterations,
            )
            current_response = refined.content

            # Re-verify
            all_chunks = list({f"{c.doc_id}_{c.chunk_index}": c
                               for c in retrieved_chunks + additional_chunks}.values())
            verification = await verifier.verify(current_response, all_chunks)

            layer_traces[f"l4_correction_{correction_iterations}"] = {
                "trust_score_after": verification.trust_score,
                "latency_ms": int((time.time() - t5) * 1000),
            }
            logger.info(
                "l4_correction_complete",
                iteration=correction_iterations,
                new_trust_score=verification.trust_score,
            )

        # ── Build Sources ────────────────────────────────────────────────
        sources = [
            SourceCitation(
                document_id=c.doc_id,
                filename=c.filename,
                page=c.page,
                excerpt=c.content[:250] + "..." if len(c.content) > 250 else c.content,
                relevance_score=c.relevance_score,
                chunk_index=c.chunk_index,
                url=c.metadata.get("url") if c.metadata else None,
                domain=c.metadata.get("domain") if c.metadata else None,
                is_web=bool(c.metadata.get("is_web", False)) if c.metadata else False,
            )
            for c in retrieved_chunks[:6]
        ]

        total_ms = int((time.time() - start_time) * 1000)
        logger.info(
            "hbia_pipeline_complete",
            trust_score=verification.trust_score,
            corrections=correction_iterations,
            latency_ms=total_ms,
        )

        return HBIAResponse(
            response=current_response,
            trust_score=verification.trust_score,
            verification=verification,
            sources=sources,
            correction_iterations=correction_iterations,
            latency_ms=total_ms,
            model_used=gen_result.model_used,
            token_usage={
                "prompt_tokens": gen_result.prompt_tokens,
                "completion_tokens": gen_result.completion_tokens,
            },
            intent=intent.intent_type,
            layer_traces=layer_traces,
            nlp_analysis=nlp.to_dict(),
        )

    async def stream_process(
        self, query: str, chat_history: Optional[List[dict]] = None, db_session = None, db_session_id = None
    ) -> AsyncGenerator[str, None]:
        """Streaming version: yields SSE-formatted events."""
        chat_history = chat_history or []

        yield 'data: {"type":"status","content":"Analysing input..."}\n\n'

        # NLP Analysis
        nlp = nlp_analyzer.analyze(query)
        import json as _json
        yield f'data: {_json.dumps({"type":"nlp_analysis","data":nlp.to_dict()})}\n\n'

        yield 'data: {"type":"status","content":"Classifying intent..."}\n\n'

        intent = await self.intent_classifier.classify(query, nlp_analysis=nlp)
        safety = await content_filter.check_input(query)

        if not safety.is_safe:
            yield f'data: {{"type":"error","content":"Request blocked: {safety.violations[0]}"}}\n\n'
            return

        yield 'data: {"type":"status","content":"Retrieving sources..."}\n\n'

        retrieved_chunks = []
        if intent.retrieval_strategy != "none" and intent.retrieval_strategy is not None:
            retrieved_chunks = await rag_retriever.retrieve(query=query, strategy=intent.retrieval_strategy)

        yield f'data: {{"type":"status","content":"Found {len(retrieved_chunks)} sources. Generating response..."}}\n\n'

        messages = prompt_builder.build_rag_prompt(query, retrieved_chunks, chat_history)

        # Stream tokens
        full_response = ""
        async for token in llm_client.generate_stream(messages, settings.default_model, 0.3, 2000):
            full_response += token
            import json as _json
            yield f'data: {_json.dumps({"type":"token","content":token})}\n\n'

        yield 'data: {"type":"status","content":"Verifying response..."}\n\n'

        verification = await verifier.verify(full_response, retrieved_chunks)
        sources = [
            {"filename": c.filename, "relevance_score": c.relevance_score, "excerpt": c.content[:200]}
            for c in retrieved_chunks[:5]
        ]

        import json as _json
        yield f'data: {_json.dumps({"type":"verification_complete","data":{"trust_score":verification.trust_score,"risk":verification.hallucination_risk,"sources":sources}})}\n\n'

        # Save to DB if provided
        if db_session and db_session_id:
            from app.db.models import Message
            import uuid
            msg_id = str(uuid.uuid4())
            assistant_msg = Message(
                id=msg_id,
                session_id=db_session_id,
                role="assistant",
                content=full_response,
                trust_score=verification.trust_score,
                sources=_json.dumps(sources),
                verification_data=_json.dumps(verification.model_dump())
            )
            db_session.add(assistant_msg)
            await db_session.commit()
            yield f'data: {_json.dumps({"type":"message_id","content":msg_id})}\n\n'

        yield 'data: {"type":"done"}\n\n'

    def _blocked_response(self, reason: str, start_time: float, layer_traces: dict) -> HBIAResponse:
        from app.layers.l3_verification.verifier import VerificationOutput
        return HBIAResponse(
            response=f"I'm unable to process this request: {reason}",
            trust_score=0.0,
            verification=VerificationOutput(trust_score=0.0, hallucination_risk="high"),
            sources=[],
            correction_iterations=0,
            latency_ms=int((time.time() - start_time) * 1000),
            model_used="blocked",
            token_usage={},
            layer_traces=layer_traces,
        )


orchestrator = HBIAOrchestrator()
