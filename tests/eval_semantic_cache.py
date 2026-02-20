"""
Semantic Cache Evaluation Script
================================
Evaluates the quality and performance of the Semantic Cache by:
1. Performance Metrics: Hit Rate, Latency Savings
2. Quality Metrics  : Cache Precision (Binary 0/1) via LLM-as-a-Judge
3. Logging          : Results → Langfuse traces

Judge Model : deepseek-reasoner (DeepSeek-V3.2 Thinking Mode)
Scoring     : Binary (0 = wrong, 1 = correct)

Usage:
    python -m tests.eval_semantic_cache
"""

import time
import os
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `src.*` and `config.*` imports work
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from src.cache.semantic_cache import semantic_cache
from src.core.state import ResearchResult
from src.utils.logger import logger


# ═══════════════════════════════════════════════════════════════════════════
# 1. TEST DATA
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CacheTestCase:
    """A single test case for cache evaluation."""
    query: str
    expected_hit: bool
    description: str
    expected_content_keywords: List[str] = field(default_factory=list)

# Estimated average cost of a full Deep Research run (Miss)
# Based on observations: ~$0.04 - $0.06 per query
ESTIMATED_AVG_COST_PER_QUERY = 0.05


# --- Seed Data (populated into cache before tests) ---
SEED_DATA = [
    ResearchResult(
        query="What are the latest advances in LLM reasoning?",
        content="Recent advances in LLM reasoning include Chain-of-Thought (CoT) prompting, "
                "Tree-of-Thought (ToT) exploration, and reinforcement learning from human feedback (RLHF). "
                "DeepSeek-R1 and OpenAI o1 are notable models that demonstrate strong multi-step reasoning.",
        source="arxiv:2401.12345",
        type="final_answer",
        confidence=1.0
    ),
    ResearchResult(
        query="How does semantic caching work in AI systems?",
        content="Semantic caching uses vector embeddings to find similar queries and return cached results "
                "instead of re-computing. It typically employs cosine similarity with a distance threshold, "
                "combined with fuzzy string matching for exact-match acceleration. Redis with RediSearch "
                "is a popular backend for this pattern.",
        source="arxiv:2403.67890",
        type="final_answer",
        confidence=1.0
    ),
    ResearchResult(
        query="What is retrieval augmented generation?",
        content="Retrieval Augmented Generation (RAG) is a technique that combines information retrieval "
                "with language generation. The system first retrieves relevant documents from a knowledge base, "
                "then uses them as context for the LLM to generate accurate, grounded responses. "
                "This reduces hallucination and improves factual accuracy.",
        source="arxiv:2402.11111",
        type="final_answer",
        confidence=1.0
    ),
]

# --- Test Cases ---
TEST_CASES = [
    # ✅ Exact match — should HIT
    CacheTestCase(
        query="What are the latest advances in LLM reasoning?",
        expected_hit=True,
        description="Exact match (identical query)",
        expected_content_keywords=["Chain-of-Thought", "reasoning"]
    ),
    # ✅ Semantic match — should HIT
    CacheTestCase(
        query="Recent breakthroughs in large language model reasoning capabilities",
        expected_hit=True,
        description="Semantic match (similar meaning, different wording)",
        expected_content_keywords=["reasoning", "CoT"]
    ),
    # ✅ Semantic match — should HIT
    CacheTestCase(
        query="How does semantic cache work?",
        expected_hit=True,
        description="Semantic match (shorter version of cached query)",
        expected_content_keywords=["vector", "embeddings"]
    ),
    # ✅ Semantic match — should HIT
    CacheTestCase(
        query="Explain RAG technique in AI",
        expected_hit=True,
        description="Semantic match (paraphrased RAG query)",
        expected_content_keywords=["retrieval", "generation"]
    ),
    # ❌ Miss — completely unrelated
    CacheTestCase(
        query="Best restaurants in Bangkok for Thai food",
        expected_hit=False,
        description="Miss (unrelated topic — food)",
    ),
    # ❌ Miss — different domain
    CacheTestCase(
        query="How to train for a marathon in 12 weeks?",
        expected_hit=False,
        description="Miss (unrelated topic — fitness)",
    ),
    # ❌ Miss — tangentially related but different intent
    CacheTestCase(
        query="How much does the DeepSeek API cost per token?",
        expected_hit=False,
        description="Miss (pricing, not technical explanation)",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# 2. PERFORMANCE METRICS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EvalResult:
    """Result from evaluating a single test case."""
    query: str
    description: str
    expected_hit: bool
    actual_hit: bool
    latency_ms: float
    cached_content: Optional[str] = None
    precision_score: Optional[int] = None  # 0 or 1 (Binary)
    judge_reason: Optional[str] = None
    cost_savings: float = 0.0


def seed_cache():
    """Populate the cache with seed data."""
    print("\n🌱 Seeding cache with test data...")
    for item in SEED_DATA:
        semantic_cache.set(item)
        print(f"   ✅ Cached: '{item.query[:60]}...'")
    print(f"   Total: {len(SEED_DATA)} items seeded.\n")


def run_performance_eval() -> List[EvalResult]:
    """Run all test cases and measure hit/miss + latency."""
    results: List[EvalResult] = []

    print("🏃 Running Performance Evaluation...")
    print("=" * 70)

    for i, tc in enumerate(TEST_CASES, 1):
        start = time.perf_counter()
        cache_result = semantic_cache.get(tc.query, filter_type="final_answer")
        elapsed_ms = (time.perf_counter() - start) * 1000

        actual_hit = cache_result is not None
        cached_content = cache_result[0].content if cache_result else None

        status = "✅ HIT" if actual_hit else "❌ MISS"
        match = "✔" if actual_hit == tc.expected_hit else "✘ UNEXPECTED"

        print(f"  [{i}/{len(TEST_CASES)}] {status} ({elapsed_ms:6.1f}ms) [{match}] {tc.description}")

        results.append(EvalResult(
            query=tc.query,
            description=tc.description,
            expected_hit=tc.expected_hit,
            actual_hit=actual_hit,
            latency_ms=elapsed_ms,
            cached_content=cached_content,
            cost_savings=ESTIMATED_AVG_COST_PER_QUERY if actual_hit else 0.0,
        ))

    return results


def print_performance_summary(results: List[EvalResult]):
    """Print aggregate performance metrics."""
    total = len(results)
    actual_hits = sum(1 for r in results if r.actual_hit)
    expected_hits = sum(1 for r in results if r.expected_hit)
    correct_predictions = sum(1 for r in results if r.actual_hit == r.expected_hit)

    hit_latencies = [r.latency_ms for r in results if r.actual_hit]
    miss_latencies = [r.latency_ms for r in results if not r.actual_hit]

    avg_hit_lat = sum(hit_latencies) / len(hit_latencies) if hit_latencies else 0
    avg_miss_lat = sum(miss_latencies) / len(miss_latencies) if miss_latencies else 0

    print("\n" + "=" * 70)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"  Total Test Cases     : {total}")
    print(f"  Expected Hits/Misses : {expected_hits} / {total - expected_hits}")
    print(f"  Actual Hits/Misses   : {actual_hits} / {total - actual_hits}")
    print(f"  Hit Rate             : {actual_hits / total * 100:.1f}%")
    print(f"  Prediction Accuracy  : {correct_predictions / total * 100:.1f}%")
    print(f"  Avg Latency (Hit)    : {avg_hit_lat:.1f}ms")
    print(f"  Avg Latency (Miss)   : {avg_miss_lat:.1f}ms")
    if avg_miss_lat > 0:
        print(f"  Latency Savings      : {avg_miss_lat - avg_hit_lat:.1f}ms per hit")
    
    total_savings = sum(r.cost_savings for r in results)
    print(f"  Est. Cost Savings    : ${total_savings:.4f} (based on ${ESTIMATED_AVG_COST_PER_QUERY}/query)")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# 3. QUALITY EVALUATION (LLM-as-a-Judge) — Binary 0/1
# ═══════════════════════════════════════════════════════════════════════════

JUDGE_PROMPT = """You are an expert evaluation judge. Your task is to determine if a cached response correctly and relevantly answers a user's query.

## User Query
{query}

## Cached Response
{cached_content}

## Instructions
1. Read the user query carefully.
2. Read the cached response.
3. Determine: Does the cached response **accurately and relevantly** answer the query?
   - If YES → score = 1
   - If NO (irrelevant, misleading, or wrong topic) → score = 0

Think step by step before giving your final answer.

## Output Format (strictly follow this)
Reasoning: <your step-by-step analysis>
Score: <0 or 1>"""


def run_quality_eval(results: List[EvalResult]) -> List[EvalResult]:
    """Use LLM-as-a-Judge (deepseek-reasoner) to evaluate cache hit quality."""
    hits = [r for r in results if r.actual_hit]
    if not hits:
        print("\n⚠️  No cache hits to evaluate quality for.")
        return results

    print(f"\n🧑‍⚖️ Running Quality Evaluation (Judge: deepseek-reasoner)...")
    print(f"   Evaluating {len(hits)} cache hits...")

    try:
        from langchain_deepseek import ChatDeepSeek
        from config.settings import settings

        # Load judge config from model_config.yaml
        judge_config = settings.model_config_yaml.get("judge_model", {})
        judge_name = judge_config.get("name", "deepseek-reasoner")
        judge_api_base = judge_config.get("api_base", "https://api.deepseek.com")

        judge_llm = ChatDeepSeek(
            model=judge_name,
            temperature=judge_config.get("temperature", 0.0),
            max_tokens=judge_config.get("max_tokens", 2048),
            api_key=settings.DEEPSEEK_API_KEY,
            api_base=judge_api_base
        )

        print(f"   Judge Model: {judge_name} (from config/model_config.yaml)")

        for i, result in enumerate(hits, 1):
            prompt = JUDGE_PROMPT.format(
                query=result.query,
                cached_content=result.cached_content
            )

            try:
                response = judge_llm.invoke(prompt)
                response_text = response.content

                # Parse score from response
                score = _parse_binary_score(response_text)
                reason = _parse_reasoning(response_text)

                result.precision_score = score
                result.judge_reason = reason

                emoji = "✅" if score == 1 else "❌"
                print(f"   [{i}/{len(hits)}] {emoji} Score={score} | {result.description}")
                if reason:
                    print(f"           Reason: {reason[:100]}...")

            except Exception as e:
                logger.error(f"Judge evaluation failed for '{result.query[:50]}': {e}")
                result.precision_score = None
                result.judge_reason = f"Error: {str(e)}"
                print(f"   [{i}/{len(hits)}] ⚠️  Error: {e}")

    except ImportError as e:
        print(f"\n⚠️  Could not load Judge LLM: {e}")
        print("   Skipping quality evaluation. Install langchain-deepseek.")

    return results


def _parse_binary_score(text: str) -> int:
    """Extract binary score (0 or 1) from Judge response."""
    lines = text.strip().split("\n")
    for line in reversed(lines):
        line_lower = line.strip().lower()
        if line_lower.startswith("score:"):
            score_str = line_lower.replace("score:", "").strip()
            if "1" in score_str:
                return 1
            elif "0" in score_str:
                return 0
    # Fallback: look for the last digit
    for char in reversed(text):
        if char in ("0", "1"):
            return int(char)
    return 0  # Default to fail if unparseable


def _parse_reasoning(text: str) -> Optional[str]:
    """Extract reasoning from Judge response."""
    lines = text.strip().split("\n")
    reasoning_lines = []
    capturing = False
    for line in lines:
        if line.strip().lower().startswith("reasoning:"):
            capturing = True
            rest = line.split(":", 1)[1].strip() if ":" in line else ""
            if rest:
                reasoning_lines.append(rest)
        elif line.strip().lower().startswith("score:"):
            capturing = False
        elif capturing:
            reasoning_lines.append(line.strip())
    return " ".join(reasoning_lines) if reasoning_lines else None


def print_quality_summary(results: List[EvalResult]):
    """Print quality evaluation summary."""
    scored = [r for r in results if r.precision_score is not None]
    if not scored:
        print("\n⚠️  No quality scores to summarize.")
        return

    total_scored = len(scored)
    correct = sum(1 for r in scored if r.precision_score == 1)

    print("\n" + "=" * 70)
    print("🧑‍⚖️ QUALITY SUMMARY (LLM-as-a-Judge: deepseek-reasoner)")
    print("=" * 70)
    print(f"  Cache Hits Evaluated : {total_scored}")
    print(f"  Precision (Correct)  : {correct}/{total_scored} ({correct / total_scored * 100:.1f}%)")
    print(f"  Scoring Method       : Binary (0=wrong, 1=correct)")
    print(f"  Judge Model          : deepseek-reasoner (V3.2 Thinking Mode)")
    print("=" * 70)


# Estimated average running time of a full Deep Research run (Miss)
ESTIMATED_AVG_TIME_PER_QUERY = 45.0

# ═══════════════════════════════════════════════════════════════════════════
# 4. LANGFUSE LOGGING
# ═══════════════════════════════════════════════════════════════════════════

def log_to_langfuse(results: List[EvalResult]):
    """Send evaluation results to Langfuse as scored traces (SDK v3)."""
    try:
        import langfuse

        # SDK v3: Initialize via environment variables
        # LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
        # are automatically picked up from .env
        lf_client = langfuse.get_client()

        # --- One span per test case ---
        for r in results:
            with lf_client.start_as_current_span(
                name="cache-eval-case",
                input={"query": r.query},
                output={"cached_content": r.cached_content or "MISS"},
                metadata={
                    "description": r.description,
                    "expected_hit": r.expected_hit,
                    "actual_hit": r.actual_hit,
                },
            ) as span:
                # Score: hit correctness (Prediction Accuracy)
                span.score(
                    name="hit_prediction",
                    value=1 if r.actual_hit == r.expected_hit else 0,
                    comment=f"Expected={'HIT' if r.expected_hit else 'MISS'}, "
                            f"Actual={'HIT' if r.actual_hit else 'MISS'}"
                )
                
                # Score: Cache Hit Status (Operational Metric)
                # 1 = Hit, 0 = Miss. Average of this = Hit Rate %.
                span.score(
                    name="is_cache_hit",
                    value=1 if r.actual_hit else 0,
                    comment="1=Hit, 0=Miss"
                )

                # Score: latency
                span.score(
                    name="latency_ms",
                    value=round(r.latency_ms, 1),
                    comment="Cache lookup latency"
                )

                # Score: precision (only for hits that were judged)
                if r.precision_score is not None:
                    span.score(
                        name="cache_precision",
                        value=r.precision_score,
                        comment=r.judge_reason or "No reasoning"
                    )

                # Score: Estimated Savings (Cost & Time)
                if r.actual_hit:
                    span.score(
                        name="cost_savings_usd",
                        value=r.cost_savings,
                        comment=f"Estimated savings based on avg run cost ${ESTIMATED_AVG_COST_PER_QUERY}"
                    )
                    span.score(
                        name="time_savings_sec",
                        value=ESTIMATED_AVG_TIME_PER_QUERY,
                        comment=f"Estimated time saved based on avg run time {ESTIMATED_AVG_TIME_PER_QUERY}s"
                    )

        lf_client.flush()
        print(f"\n📡 Logged {len(results)} traces to Langfuse ✅")

    except ImportError:
        print("\n⚠️  Langfuse not installed. Skipping logging.")
    except Exception as e:
        print(f"\n⚠️  Langfuse logging failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   🔬 Semantic Cache Evaluation                             ║")
    print("║   Judge : deepseek-reasoner (V3.2 Thinking Mode)           ║")
    print("║   Score : Binary (0/1)                                     ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Step 1: Seed cache
    seed_cache()

    # Step 2: Performance evaluation
    results = run_performance_eval()
    print_performance_summary(results)

    # Step 3: Quality evaluation (LLM-as-a-Judge)
    results = run_quality_eval(results)
    print_quality_summary(results)

    # Step 4: Log to Langfuse
    log_to_langfuse(results)

    print("\n✅ Evaluation complete!\n")


if __name__ == "__main__":
    main()
