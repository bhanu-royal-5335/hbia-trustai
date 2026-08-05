"""
Layer 0.5: NLP Input Analyser
Performs lightweight, zero-dependency Natural Language Processing on every user
query to extract structured insights that feed into intent classification,
prompt building, and the UI.

Techniques used (no external libraries required):
  - Tokenisation          — regex-based word/sentence splitting
  - Named Entity Recognition — regex patterns + curated gazetteers
  - Keyword Extraction    — TF-IDF–style scoring against a stopword list
  - Sentiment Analysis    — AFINN-style lexicon scoring
  - Topic Detection       — keyword → taxonomy mapping
  - Complexity Scoring    — word count, clause depth, question type
  - Language Detection    — character n-gram fingerprinting (basic)
"""
from __future__ import annotations

import re
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import structlog

logger = structlog.get_logger()

# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectedEntity:
    text: str
    entity_type: str   # PERSON | ORG | LOCATION | DATE | TECH | CONCEPT
    start: int
    end: int
    confidence: float


@dataclass
class NLPAnalysis:
    # Core analysis
    tokens: List[str] = field(default_factory=list)
    sentences: List[str] = field(default_factory=list)
    entities: List[DetectedEntity] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    sentiment: str = "neutral"          # positive | negative | neutral
    sentiment_score: float = 0.0        # -1.0 to +1.0
    topics: List[str] = field(default_factory=list)
    complexity: str = "moderate"        # simple | moderate | complex
    complexity_score: float = 0.5       # 0.0 to 1.0
    language: str = "en"
    word_count: int = 0
    question_type: Optional[str] = None # what | who | when | where | how | why | yes_no | none
    intent_hints: List[str] = field(default_factory=list)   # soft signals for L0 classifier

    def to_dict(self) -> dict:
        return {
            "tokens": self.tokens[:20],  # cap for API payload
            "sentences": self.sentences,
            "entities": [
                {
                    "text": e.text,
                    "type": e.entity_type,
                    "confidence": round(e.confidence, 2),
                }
                for e in self.entities
            ],
            "keywords": self.keywords[:10],
            "sentiment": self.sentiment,
            "sentiment_score": round(self.sentiment_score, 3),
            "topics": self.topics,
            "complexity": self.complexity,
            "complexity_score": round(self.complexity_score, 2),
            "language": self.language,
            "word_count": self.word_count,
            "question_type": self.question_type,
            "intent_hints": self.intent_hints,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Stopwords
# ─────────────────────────────────────────────────────────────────────────────

STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "shall", "should", "may", "might", "must", "can", "could", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "up", "about", "into", "through",
    "during", "before", "after", "above", "below", "between", "out", "off",
    "over", "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "not", "only", "own", "same", "so", "than",
    "too", "very", "just", "i", "me", "my", "we", "our", "ours", "you", "your",
    "he", "she", "it", "they", "them", "this", "that", "these", "those", "what",
    "which", "who", "whom", "its", "if", "as", "s", "t", "don", "it", "give",
    "tell", "let", "get", "make", "know", "think", "see", "go", "come", "take",
    "find", "look", "want", "use", "need", "said", "say", "also", "now", "new",
    "like", "please", "help", "can", "could", "would", "much", "many", "well",
})

# ─────────────────────────────────────────────────────────────────────────────
# Named Entity Gazetteers & Patterns
# ─────────────────────────────────────────────────────────────────────────────

# Technology terms
TECH_TERMS = frozenset({
    "python", "javascript", "typescript", "java", "c++", "rust", "go", "kotlin",
    "swift", "ruby", "php", "sql", "nosql", "html", "css", "react", "vue",
    "angular", "nextjs", "fastapi", "django", "flask", "node", "nodejs", "docker",
    "kubernetes", "aws", "azure", "gcp", "tensorflow", "pytorch", "scikit", "pandas",
    "numpy", "langchain", "openai", "gpt", "chatgpt", "claude", "gemini", "llm",
    "ai", "ml", "nlp", "deep learning", "machine learning", "neural", "api",
    "rest", "graphql", "git", "github", "linux", "unix", "windows", "macos",
    "blockchain", "bitcoin", "ethereum", "crypto", "mongodb", "postgresql",
    "mysql", "redis", "elasticsearch", "kafka", "spark", "hadoop", "tensorflow",
    "cuda", "gpu", "cpu", "ram", "ssd", "cloud", "microservices", "devops",
    "ci/cd", "jenkins", "terraform", "ansible", "prometheus", "grafana",
    "vector", "embedding", "rag", "fine-tuning", "transformer", "bert", "gpt-4",
    "hugging face", "chromadb", "pinecone", "weaviate", "langchain", "llamaindex",
})

# Location indicators
LOCATION_PATTERNS = [
    r'\b(?:in|at|from|near|to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
    r'\b(Silicon Valley|New York|Los Angeles|San Francisco|Washington DC|'
     r'United States|United Kingdom|India|China|Europe|Asia|Africa|Australia|'
     r'Canada|Germany|France|Japan|South Korea|Brazil|Russia|Middle East)\b',
]

# Date / time patterns
DATE_PATTERNS = [
    r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s*\d{4})?\b',
    r'\b\d{4}\b',  # Year
    r'\b(?:yesterday|today|tomorrow|last\s+(?:week|month|year)|next\s+(?:week|month|year)|recently|currently)\b',
    r'\b(?:in\s+the\s+(?:19|20)\d{2}s)\b',
]

# Organisation prefix words
ORG_PREFIXES = frozenset({
    "company", "corporation", "corp", "inc", "ltd", "llc", "university",
    "institute", "organization", "foundation", "agency", "department",
    "ministry", "government", "committee", "association", "union", "group",
    "enterprises", "industries", "solutions", "systems", "technologies",
    "lab", "labs", "research",
})

# ─────────────────────────────────────────────────────────────────────────────
# Sentiment Lexicon  (AFINN-inspired, hand-curated subset)
# ─────────────────────────────────────────────────────────────────────────────

SENTIMENT_LEXICON: Dict[str, float] = {
    # Strong positive
    "excellent": 2.0, "amazing": 2.0, "outstanding": 2.0, "fantastic": 2.0,
    "superb": 2.0, "brilliant": 1.8, "wonderful": 1.8, "perfect": 1.8,
    "love": 1.5, "great": 1.5, "best": 1.5, "good": 1.0, "nice": 1.0,
    "helpful": 1.0, "useful": 1.0, "correct": 0.8, "right": 0.7,
    "clear": 0.7, "fast": 0.6, "efficient": 0.8, "easy": 0.6,
    "innovative": 1.2, "powerful": 1.0, "impressive": 1.5, "beautiful": 1.2,
    "elegant": 1.0, "robust": 0.8, "reliable": 0.8, "accurate": 0.9,
    # Mild positive
    "ok": 0.3, "okay": 0.3, "fine": 0.3, "decent": 0.4,
    # Negative
    "bad": -1.0, "terrible": -2.0, "awful": -2.0, "horrible": -2.0,
    "poor": -1.2, "wrong": -1.0, "slow": -0.8, "broken": -1.5,
    "fail": -1.5, "failed": -1.5, "failure": -1.5, "error": -0.8,
    "bug": -0.7, "issue": -0.5, "problem": -0.8, "hate": -2.0,
    "dislike": -1.2, "difficult": -0.5, "hard": -0.4, "confusing": -0.9,
    "complicated": -0.6, "useless": -1.8, "waste": -1.5, "deprecated": -0.7,
    "outdated": -0.8, "ugly": -1.2, "frustrating": -1.5, "annoying": -1.2,
    "disappointed": -1.5, "disappointing": -1.5, "worst": -2.0,
    # Negation boosters — handled separately
    "not": 0, "no": 0, "never": 0, "neither": 0,
}

NEGATION_WORDS = frozenset({"not", "no", "never", "neither", "nor", "without", "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't"})

# ─────────────────────────────────────────────────────────────────────────────
# Topic Taxonomy
# ─────────────────────────────────────────────────────────────────────────────

TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "Technology": [
        "software", "hardware", "code", "program", "api", "database", "server",
        "app", "web", "mobile", "cloud", "network", "security", "cyber",
        "algorithm", "data", "computer", "internet", "digital", "system",
        "linux", "windows", "macos", "devops", "automation", "script",
        "framework", "library", "deployment", "backend", "frontend", "fullstack",
    ],
    "Artificial Intelligence": [
        "ai", "machine learning", "deep learning", "neural", "llm", "model",
        "training", "inference", "nlp", "computer vision", "gpt", "bert",
        "transformer", "embedding", "vector", "rag", "chatbot", "language model",
        "generative", "prompt", "fine-tuning", "openai", "anthropic", "gemini",
        "hugging face", "tensorflow", "pytorch", "scikit",
    ],
    "Science": [
        "research", "experiment", "hypothesis", "theory", "biology", "chemistry",
        "physics", "astronomy", "genetics", "molecule", "atom", "quantum",
        "evolution", "climate", "ecology", "neuroscience", "psychology",
        "medicine", "disease", "vaccine", "dna", "protein", "cell",
    ],
    "Health & Medicine": [
        "health", "medical", "disease", "treatment", "drug", "medicine",
        "symptom", "diagnosis", "therapy", "doctor", "hospital", "patient",
        "surgery", "vaccine", "virus", "cancer", "diabetes", "mental health",
        "nutrition", "exercise", "fitness", "wellness", "pharmacy",
    ],
    "Finance & Business": [
        "finance", "money", "investment", "stock", "market", "economy",
        "business", "startup", "revenue", "profit", "loss", "tax", "bank",
        "crypto", "bitcoin", "trading", "budget", "cost", "price", "salary",
        "funding", "vc", "venture", "equity", "debt", "loan",
    ],
    "Law & Policy": [
        "law", "legal", "court", "regulation", "policy", "government", "rights",
        "constitution", "act", "bill", "legislation", "attorney", "judge",
        "privacy", "gdpr", "compliance", "contract", "patent", "copyright",
        "lawsuit", "crime", "criminal", "civil",
    ],
    "History": [
        "history", "historical", "ancient", "war", "civilization", "century",
        "empire", "revolution", "dynasty", "archaeology", "culture", "tradition",
        "era", "period", "timeline", "event", "biography",
    ],
    "Mathematics": [
        "math", "algebra", "calculus", "statistics", "probability", "geometry",
        "equation", "formula", "proof", "theorem", "matrix", "vector",
        "integral", "derivative", "function", "graph", "number theory",
    ],
    "Education": [
        "education", "school", "university", "college", "degree", "course",
        "study", "learning", "teacher", "student", "exam", "curriculum",
        "scholarship", "academic", "tutor",
    ],
    "Sports & Entertainment": [
        "sport", "football", "soccer", "basketball", "cricket", "tennis",
        "game", "movie", "music", "film", "actor", "artist", "album",
        "series", "show", "entertainment", "celebrity", "award",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Intent Hint Signals
# ─────────────────────────────────────────────────────────────────────────────

CODE_SIGNALS = frozenset({
    "write", "code", "script", "function", "program", "implement", "debug",
    "fix", "error", "traceback", "compile", "run", "execute", "class", "method",
    "loop", "recursion", "algorithm", "snippet", "example", "syntax",
})

SUMMARISE_SIGNALS = frozenset({
    "summarize", "summarise", "summary", "tldr", "brief", "overview",
    "in short", "key points", "main points", "gist",
})

COMPARE_SIGNALS = frozenset({
    "compare", "difference", "versus", "vs", "better", "worse", "pros",
    "cons", "advantages", "disadvantages", "trade-off", "contrast",
})

CREATIVE_SIGNALS = frozenset({
    "write a story", "poem", "essay", "creative", "fiction", "novel",
    "brainstorm", "ideas for", "generate ideas", "suggest",
})

CONVERSATIONAL_SIGNALS = frozenset({
    "hello", "hi", "hey", "thanks", "thank you", "goodbye", "bye",
    "how are you", "what's up", "good morning", "good evening",
})


# ─────────────────────────────────────────────────────────────────────────────
# NLP Analyser Class
# ─────────────────────────────────────────────────────────────────────────────

class NLPAnalyzer:
    """Lightweight, zero-dependency NLP analysis engine."""

    # IDF weights simulated from a generic large corpus
    _IDF_BASE = 10.0

    def analyze(self, text: str) -> NLPAnalysis:
        """Run full NLP pipeline on the input text and return NLPAnalysis."""
        if not text or not text.strip():
            return NLPAnalysis()

        try:
            tokens = self._tokenize(text)
            sentences = self._split_sentences(text)
            entities = self._extract_entities(text, tokens)
            keywords = self._extract_keywords(tokens, entities)
            sentiment, sentiment_score = self._analyze_sentiment(tokens)
            topics = self._detect_topics(tokens, entities)
            complexity, complexity_score = self._score_complexity(text, tokens, sentences)
            language = self._detect_language(text)
            question_type = self._classify_question(text, tokens)
            intent_hints = self._build_intent_hints(tokens, topics, entities, question_type)

            result = NLPAnalysis(
                tokens=tokens,
                sentences=sentences,
                entities=entities,
                keywords=keywords,
                sentiment=sentiment,
                sentiment_score=sentiment_score,
                topics=topics,
                complexity=complexity,
                complexity_score=complexity_score,
                language=language,
                word_count=len(tokens),
                question_type=question_type,
                intent_hints=intent_hints,
            )

            logger.info(
                "nlp_analysis_complete",
                word_count=len(tokens),
                entities=len(entities),
                keywords=keywords[:5],
                sentiment=sentiment,
                topics=topics,
                complexity=complexity,
                question_type=question_type,
                intent_hints=intent_hints,
            )
            return result

        except Exception as e:
            logger.warning("nlp_analysis_failed", error=str(e))
            return NLPAnalysis(
                tokens=text.lower().split(),
                word_count=len(text.split()),
            )

    # ── Tokenisation ──────────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> List[str]:
        """Split text into lowercase word tokens, removing punctuation."""
        words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9'\-]*\b", text.lower())
        return [w for w in words if len(w) >= 2]

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    # ── Named Entity Recognition ──────────────────────────────────────────────

    def _extract_entities(self, text: str, tokens: List[str]) -> List[DetectedEntity]:
        entities: List[DetectedEntity] = []
        seen_spans: set = set()

        # --- TECH entities ---
        for tech in TECH_TERMS:
            pattern = r'\b' + re.escape(tech) + r'\b'
            for m in re.finditer(pattern, text, re.IGNORECASE):
                span_key = (m.start(), m.end())
                if span_key not in seen_spans:
                    seen_spans.add(span_key)
                    entities.append(DetectedEntity(
                        text=m.group(),
                        entity_type="TECH",
                        start=m.start(),
                        end=m.end(),
                        confidence=0.92,
                    ))

        # --- PERSON entities: capitalised full names (2+ words) ---
        for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text):
            name = m.group()
            span_key = (m.start(), m.end())
            # Exclude pure location / org patterns caught later
            if span_key not in seen_spans and len(name.split()) <= 4:
                seen_spans.add(span_key)
                entities.append(DetectedEntity(
                    text=name,
                    entity_type="PERSON",
                    start=m.start(),
                    end=m.end(),
                    confidence=0.72,
                ))

        # --- LOCATION entities ---
        for pattern in LOCATION_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                loc = m.group(1) if m.lastindex else m.group()
                span_key = (m.start(), m.end())
                if span_key not in seen_spans and loc:
                    seen_spans.add(span_key)
                    entities.append(DetectedEntity(
                        text=loc,
                        entity_type="LOCATION",
                        start=m.start(),
                        end=m.end(),
                        confidence=0.80,
                    ))

        # --- DATE entities ---
        for pattern in DATE_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                span_key = (m.start(), m.end())
                if span_key not in seen_spans:
                    seen_spans.add(span_key)
                    entities.append(DetectedEntity(
                        text=m.group(),
                        entity_type="DATE",
                        start=m.start(),
                        end=m.end(),
                        confidence=0.85,
                    ))

        # --- ORG entities: capitalised tokens followed by org suffixes ---
        for m in re.finditer(
            r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(' +
            '|'.join(re.escape(p) for p in sorted(ORG_PREFIXES, key=len, reverse=True)) +
            r')\b',
            text, re.IGNORECASE
        ):
            span_key = (m.start(), m.end())
            if span_key not in seen_spans:
                seen_spans.add(span_key)
                entities.append(DetectedEntity(
                    text=m.group(),
                    entity_type="ORG",
                    start=m.start(),
                    end=m.end(),
                    confidence=0.78,
                ))

        # Deduplicate by text (keep highest confidence)
        deduped: Dict[str, DetectedEntity] = {}
        for e in entities:
            key = e.text.lower()
            if key not in deduped or e.confidence > deduped[key].confidence:
                deduped[key] = e

        return sorted(deduped.values(), key=lambda x: x.start)

    # ── Keyword Extraction ────────────────────────────────────────────────────

    def _extract_keywords(
        self, tokens: List[str], entities: List[DetectedEntity]
    ) -> List[str]:
        """TF-IDF–inspired keyword extraction."""
        content_tokens = [t for t in tokens if t not in STOPWORDS and len(t) >= 3]
        if not content_tokens:
            return []

        tf: Counter = Counter(content_tokens)
        total = len(content_tokens)

        # Boost entity tokens
        entity_words = set()
        for e in entities:
            for w in e.text.lower().split():
                entity_words.add(w)

        scores: Dict[str, float] = {}
        for word, count in tf.items():
            tf_score = count / total
            # Inverse doc frequency approximation: rarer tokens score higher
            idf = math.log(self._IDF_BASE / (1 + count))
            boost = 1.5 if word in entity_words else 1.0
            scores[word] = tf_score * max(idf, 0.1) * boost

        # Also include bigrams
        bigrams = self._extract_bigrams(content_tokens)
        for bigram, bcount in bigrams.items():
            bg_score = (bcount / total) * math.log(self._IDF_BASE / (1 + bcount)) * 1.8
            if bg_score > 0:
                scores[bigram] = bg_score

        sorted_kw = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [kw for kw, _ in sorted_kw[:12]]

    def _extract_bigrams(self, tokens: List[str]) -> Counter:
        bigrams: Counter = Counter()
        for i in range(len(tokens) - 1):
            if tokens[i] not in STOPWORDS and tokens[i + 1] not in STOPWORDS:
                bigram = f"{tokens[i]} {tokens[i + 1]}"
                bigrams[bigram] += 1
        return bigrams

    # ── Sentiment Analysis ────────────────────────────────────────────────────

    def _analyze_sentiment(self, tokens: List[str]) -> Tuple[str, float]:
        """Lexicon-based sentiment with negation handling."""
        score = 0.0
        negate = False
        negate_window = 0

        for token in tokens:
            if token in NEGATION_WORDS:
                negate = True
                negate_window = 3
                continue

            if negate_window > 0:
                negate_window -= 1
                if negate_window == 0:
                    negate = False

            if token in SENTIMENT_LEXICON:
                val = SENTIMENT_LEXICON[token]
                score += (-val * 0.8) if negate else val

        # Normalise to [-1, 1]
        norm_score = max(-1.0, min(1.0, score / max(len(tokens), 1) * 5))

        if norm_score >= 0.1:
            sentiment = "positive"
        elif norm_score <= -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return sentiment, round(norm_score, 3)

    # ── Topic Detection ───────────────────────────────────────────────────────

    def _detect_topics(
        self, tokens: List[str], entities: List[DetectedEntity]
    ) -> List[str]:
        """Map query tokens to predefined topic taxonomy."""
        token_set = set(tokens)
        text_lower = " ".join(tokens)

        topic_scores: Dict[str, float] = {}
        for topic, keywords in TOPIC_KEYWORDS.items():
            score = 0.0
            for kw in keywords:
                if " " in kw:  # multi-word keyword
                    if kw in text_lower:
                        score += 2.0
                elif kw in token_set:
                    score += 1.0
            if score > 0:
                topic_scores[topic] = score

        # Boost AI/Tech topics when tech entities present
        tech_entities = [e for e in entities if e.entity_type == "TECH"]
        if tech_entities:
            topic_scores["Technology"] = topic_scores.get("Technology", 0) + len(tech_entities) * 0.8
            topic_scores["Artificial Intelligence"] = (
                topic_scores.get("Artificial Intelligence", 0) + 0.5
            )

        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        return [t for t, s in sorted_topics[:4] if s >= 1.0]

    # ── Complexity Scoring ────────────────────────────────────────────────────

    def _score_complexity(
        self, text: str, tokens: List[str], sentences: List[str]
    ) -> Tuple[str, float]:
        """Heuristic complexity based on word count, sentence structure, and jargon."""
        wc = len(tokens)
        sc = len(sentences)
        avg_sentence_len = wc / max(sc, 1)

        # Count subordinating conjunctions / relative clauses
        clause_markers = len(re.findall(
            r'\b(which|that|because|although|since|while|unless|whether|if|'
            r'though|however|therefore|furthermore|moreover|nevertheless)\b',
            text, re.IGNORECASE
        ))

        # Count jargon (non-stopword tokens longer than 8 chars)
        jargon_count = sum(1 for t in tokens if len(t) > 8 and t not in STOPWORDS)

        # Compute composite score
        score = 0.0
        score += min(wc / 50, 0.4)            # length contribution
        score += min(avg_sentence_len / 25, 0.2)  # sentence length
        score += min(clause_markers / 5, 0.2)  # clause complexity
        score += min(jargon_count / 10, 0.2)   # jargon density

        score = max(0.0, min(1.0, score))

        if score < 0.35:
            label = "simple"
        elif score < 0.65:
            label = "moderate"
        else:
            label = "complex"

        return label, round(score, 2)

    # ── Language Detection ────────────────────────────────────────────────────

    def _detect_language(self, text: str) -> str:
        """Minimal n-gram–based language fingerprint (returns 'en' or 'unknown')."""
        # Very basic: count English stopword density
        tokens = text.lower().split()
        if not tokens:
            return "unknown"
        en_density = sum(1 for t in tokens if t in STOPWORDS) / len(tokens)
        return "en" if en_density >= 0.1 else "unknown"

    # ── Question Classification ───────────────────────────────────────────────

    def _classify_question(self, text: str, tokens: List[str]) -> Optional[str]:
        """Classify question type from first interrogative word."""
        first_three = tokens[:3] if tokens else []
        text_start = text.strip().lower()[:50]

        if any(w in ("what", "which") for w in first_three) or text_start.startswith(("what ", "which ")):
            return "what"
        if "who" in first_three or text_start.startswith("who "):
            return "who"
        if "when" in first_three or text_start.startswith("when "):
            return "when"
        if "where" in first_three or text_start.startswith("where "):
            return "where"
        if "why" in first_three or text_start.startswith("why "):
            return "why"
        if "how" in first_three or text_start.startswith("how "):
            return "how"
        if tokens and tokens[0] in ("is", "are", "was", "were", "do", "does", "did",
                                     "can", "could", "will", "would", "should", "shall"):
            return "yes_no"
        if "?" in text:
            return "what"   # Generic question
        return None

    # ── Intent Hint Builder ───────────────────────────────────────────────────

    def _build_intent_hints(
        self,
        tokens: List[str],
        topics: List[str],
        entities: List[DetectedEntity],
        question_type: Optional[str],
    ) -> List[str]:
        """Generate soft intent signals for the L0 intent classifier."""
        hints: List[str] = []
        token_set = set(tokens)

        # Direct signal checks
        if token_set & CODE_SIGNALS:
            hints.append("code_generation")
        if token_set & SUMMARISE_SIGNALS:
            hints.append("summarization")
        if token_set & COMPARE_SIGNALS:
            hints.append("comparison")
        if token_set & CREATIVE_SIGNALS:
            hints.append("creative")
        if token_set & CONVERSATIONAL_SIGNALS:
            hints.append("conversational")

        # Topic-based signals
        if "Artificial Intelligence" in topics or "Technology" in topics:
            hints.append("factual_query")
        if question_type in ("who", "what", "when", "where"):
            hints.append("factual_query")
        if question_type == "why" or question_type == "how":
            hints.append("question_answer")

        # Entity-based signals
        person_entities = [e for e in entities if e.entity_type == "PERSON"]
        if person_entities:
            hints.append("factual_query")

        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for h in hints:
            if h not in seen:
                seen.add(h)
                deduped.append(h)
        return deduped


# ─────────────────────────────────────────────────────────────────────────────
# Singleton instance
# ─────────────────────────────────────────────────────────────────────────────

nlp_analyzer = NLPAnalyzer()
