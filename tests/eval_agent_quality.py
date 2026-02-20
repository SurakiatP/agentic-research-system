"""
Agent Quality Evaluation Script
================================
Evaluates the Deep Research Agent's response quality using DeepEval metrics.

Metrics (from Evaluation Strategy):
1. Faithfulness       — Does the answer hallucinate? (Binary 0/1)
2. Answer Relevancy   — Does the answer address the user's query? (Binary 0/1) 
3. Tool Correctness   — Did the agent choose the right tools? (Binary 0/1)
4. Summarization      — Quality of the summarized output (Rubric 1-5)
5. Plan Quality       — Was the research plan well-designed? (G-Eval, Rubric 1-5)

Judge Model: deepseek-reasoner (V3.2 Thinking Mode)

Usage:
    python -m tests.eval_agent_quality
"""

import os
import sys
import time
import uuid
from typing import Dict, Any, List, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from src.core.graph import graph_app
from src.utils.logger import logger
from src.eval.llm import DeepSeekJudge
from config.settings import settings

# Langfuse auto-tracing for agent runs inside eval
try:
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
    _langfuse_handler = LangfuseCallbackHandler()
except Exception:
    _langfuse_handler = None

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Test queries — each tests a different aspect of the agent
TEST_QUERIES = [
    {
        "query": "What are the latest advancements in Retrieval-Augmented Generation (RAG) for 2024-2025?",
        "expected_tools": ["tavily_search", "arxiv_search"],
        "category": "deep_research",
    },
    {
        "query": "Compare the architectural differences between Mamba, RWKV, and Transformer models for long-context tasks",
        "expected_tools": ["tavily_search", "arxiv_search"],
        "category": "deep_research",
    },
    {
        "query": "What are the current approaches to reducing hallucination in Large Language Models?",
        "expected_tools": ["tavily_search", "arxiv_search"],
        "category": "deep_research",
    },
    {
        "query": "What is the Model Context Protocol (MCP) introduced by Anthropic, and what are its core capabilities?",
        "expected_tools": ["tavily_search", "brave_search"],
        "category": "deep_research",
    },
    {
        "query": "Can you summarize the latest research papers related to Q-Star (Q*) algorithms?",
        "expected_tools": ["arxiv_search"],
        "category": "deep_research",
    },
    {
        "query": "Could you provide a recent overview of the global semiconductor market trends for 2025?",
        "expected_tools": ["tavily_search"],
        "category": "deep_research",
    },
    {
        "query": "Extract and summarize the main content from https://en.wikipedia.org/wiki/Artificial_general_intelligence",
        "expected_tools": ["scrape_website", "tavily_search"],
        "category": "deep_research",
    },
    {
        "query": "Write a simple Python script to start an HTTP server.",
        "expected_tools": [],
        "category": "general_chat",
    },
    {
        "query": "Tell me a creative joke about software engineers.",
        "expected_tools": [],
        "category": "general_chat",
    },
]

# Default expected tools (fallback)
EXPECTED_TOOLS = ["tavily_search", "arxiv_search"]


# ═══════════════════════════════════════════════════════════════════════════
# AGENT RUNNER (reused pattern from eval_e2e_system.py)
# ═══════════════════════════════════════════════════════════════════════════

def run_agent_and_capture(query: str) -> Dict[str, Any]:
    """
    Runs the full agent pipeline with auto-approval at HITL breakpoints.
    Returns a dict with all data needed for evaluation:
        - actual_output: The final answer
        - retrieval_context: List of research contents
        - tools_called: List of tool names used
        - sub_queries: The research plan from Manager
        - elapsed_sec: Total execution time
    """
    thread_id = f"eval_quality_{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
    if _langfuse_handler:
        config["callbacks"] = [_langfuse_handler]
    
    print(f"   -> Starting agent for: '{query[:60]}...'")
    start = time.perf_counter()
    
    current_input = {"user_query": query}
    steps = 0
    
    while True:
        try:
            result = graph_app.invoke(current_input, config=config)
            snapshot = graph_app.get_state(config)
            next_node = snapshot.next
            
            if not next_node:
                print("     [OK] Execution finished.")
                break
            
            print(f"     [PAUSE] Interrupted at: {next_node}")
            
            if "researcher" in next_node:
                print("     [AUTO] Approving Research Plan (HITL #1)...")
                current_input = None
            elif "summarize_answer" in next_node:
                print("     [AUTO] Approving Final Answer (HITL #2)...")
                graph_app.update_state(config, {"user_approved_final": True})
                current_input = None
            else:
                print(f"     [AUTO] Resuming at {next_node}...")
                current_input = None
            
            steps += 1
            if steps > 10:
                print("     [WARN] Too many interrupts, aborting.")
                break
                
        except Exception as e:
            print(f"     [ERROR] {e}")
            break
    
    elapsed = time.perf_counter() - start
    
    # Extract data from final state
    final_state = graph_app.get_state(config)
    state_values = final_state.values if hasattr(final_state, 'values') else {}
    
    actual_output = state_values.get("final_answer", "")
    research_results = state_values.get("research_results", [])
    tools_called = state_values.get("tools_called", [])
    sub_queries = state_values.get("sub_queries", [])
    cache_hit = state_values.get("cache_hit", False)
    
    # Build retrieval_context from research_results
    retrieval_context = []
    for r in research_results:
        if hasattr(r, "content"):
            retrieval_context.append(r.content)
        elif isinstance(r, dict):
            retrieval_context.append(r.get("content", ""))
    
    # Build plan text from sub_queries
    plan_text = ""
    for sq in sub_queries:
        if hasattr(sq, "sub_query"):
            plan_text += f"- {sq.sub_query} (Reason: {sq.reasoning})\n"
        elif isinstance(sq, dict):
            plan_text += f"- {sq.get('sub_query', '')} (Reason: {sq.get('reasoning', '')})\n"
    
    # ── Cost Estimation ──
    tool_cost, tool_breakdown = _estimate_tool_cost(tools_called)
    model_cost = _estimate_model_cost(actual_output, retrieval_context)
    total_cost = tool_cost + model_cost
    
    return {
        "actual_output": actual_output or "",
        "retrieval_context": retrieval_context,
        "tools_called": tools_called,
        "plan_text": plan_text,
        "sub_queries": sub_queries,
        "elapsed_sec": elapsed,
        "cache_hit": cache_hit,
        "cost": {
            "tool_cost": tool_cost,
            "tool_breakdown": tool_breakdown,
            "model_cost": model_cost,
            "total": total_cost,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# COST ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════

# DeepSeek pricing (USD per 1M tokens, cache-miss)
_DEEPSEEK_CHAT_INPUT = 0.27 / 1_000_000
_DEEPSEEK_CHAT_OUTPUT = 1.10 / 1_000_000
_DEEPSEEK_REASONER_INPUT = 0.55 / 1_000_000
_DEEPSEEK_REASONER_OUTPUT = 2.19 / 1_000_000

# Map tool names (as called) to settings keys
_TOOL_COST_MAP = {
    "tavily_search": "tavily",
    "brave_search": "brave",
    "scrape_website": "firecrawl",
    "arxiv_search": "arxiv",
}


def _estimate_tool_cost(tools_called: list) -> tuple:
    """Estimate cost of tool calls. Returns (total, breakdown_dict)."""
    total = 0.0
    breakdown = {}
    for tool_name in tools_called:
        key = _TOOL_COST_MAP.get(tool_name, tool_name)
        c = settings.TOOL_COSTS.get(key, 0.0)
        total += c
        breakdown[key] = breakdown.get(key, 0.0) + c
    return total, breakdown


def _estimate_model_cost(output: str, context: list) -> float:
    """
    Rough estimate of DeepSeek Chat model cost for a single query run.
    Uses character count / 4 as token approximation.
    """
    # Approximate tokens: chars / 4 (rough English estimate)
    output_tokens = len(output) / 4
    context_tokens = sum(len(c) for c in context) / 4
    # Rough input tokens: context + prompts (~2K overhead per node)
    input_tokens = context_tokens + 2000 * 4  # ~4 nodes (router, manager, researcher, summarizer)
    return (input_tokens * _DEEPSEEK_CHAT_INPUT) + (output_tokens * _DEEPSEEK_CHAT_OUTPUT)


def _estimate_judge_cost(num_calls: int, avg_prompt_tokens: int = 3000) -> float:
    """Estimate cost of judge (deepseek-reasoner) calls."""
    input_cost = num_calls * avg_prompt_tokens * _DEEPSEEK_REASONER_INPUT
    output_cost = num_calls * 400 * _DEEPSEEK_REASONER_OUTPUT  # ~400 output tokens per judge call
    return input_cost + output_cost


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION (DeepEval)
# ═══════════════════════════════════════════════════════════════════════════

def build_metrics(judge: DeepSeekJudge) -> list:
    """
    Build the 5 evaluation metrics from the Evaluation Strategy.
    Returns a list of (metric_name, evaluator_fn) tuples.
    Each evaluator_fn takes (input, actual_output, retrieval_context, tools_called, plan_text)
    and returns (score: float, reason: str).
    """
    metrics = []
    
    # ── 1. Faithfulness (Binary 0/1) ──
    def eval_faithfulness(input_q, output, context, **_):
        if not context:
            return 0.0, "No retrieval context available to verify faithfulness."
        
        ctx_text = "\n---\n".join(context)
        # Increase context limit to 25,000 chars (approx 6-8k tokens) for Deep Research
        truncated_ctx = ctx_text[:25000]
        
        prompt = f"""You are an impartial judge evaluating FAITHFULNESS.

FAITHFULNESS means: Every claim in the answer MUST be supported by evidence in the retrieval context.
If even ONE claim is fabricated (not found in context), score = 0.

## User Question
{input_q}

## Answer to Evaluate
## Answer to Evaluate
{output[:5000]}

## Retrieval Context (Source Documents)
{truncated_ctx}

## Task
1. List each factual claim in the answer.
2. For each claim, check if it appears in the context.
3. If ALL claims are supported -> score = 1. If ANY claim is unsupported -> score = 0.

Respond in this exact format:
SCORE: <0 or 1>
REASON: <brief explanation>"""
        
        resp = judge.generate(prompt)
        return _parse_score_reason(resp, binary=True)
    
    metrics.append(("faithfulness", eval_faithfulness))
    
    # ── 2. Answer Relevancy (Binary 0/1) ──
    def eval_relevancy(input_q, output, **_):
        prompt = f"""You are an impartial judge evaluating ANSWER RELEVANCY.

RELEVANCY means: The answer directly addresses the user's question.
If the answer is off-topic, evasive, or mostly irrelevant -> score = 0.

## User Question
{input_q}

## Answer to Evaluate
{output[:3000]}

## Task
Does this answer directly and substantively address the user's question?
- Score 1 if the answer is relevant and addresses the question.
- Score 0 if the answer is off-topic, evasive, or irrelevant.

Respond in this exact format:
SCORE: <0 or 1>
REASON: <brief explanation>"""
        
        resp = judge.generate(prompt)
        return _parse_score_reason(resp, binary=True)
    
    metrics.append(("answer_relevancy", eval_relevancy))
    
    # ── 3. Tool Correctness (Binary 0/1) ──
    def eval_tool_correctness(input_q, tools_called, expected_tools=None, **_):
        exp = expected_tools if expected_tools is not None else EXPECTED_TOOLS
        
        # If no tools are expected (e.g., general_chat), having none is correct
        if not exp:
            if not tools_called:
                return 1.0, "Correctly used no tools (as expected for this query type)."
            else:
                return 0.0, f"Expected no tools but called {set(tools_called)}"
        
        # Check if at least one expected tool was called
        called_set = set(tools_called)
        expected_set = set(exp)
        overlap = called_set & expected_set
        
        if overlap:
            return 1.0, f"Correctly used: {', '.join(overlap)}"
        else:
            return 0.0, f"Expected {expected_set} but called {called_set}"
    
    metrics.append(("tool_correctness", eval_tool_correctness))
    
    # ── 4. Summarization (Rubric 1-5) ──
    def eval_summarization(input_q, output, context, **_):
        ctx_text = "\n---\n".join(context[:5]) if context else "N/A"
        prompt = f"""You are an impartial judge evaluating SUMMARIZATION QUALITY.

Rate the summary on a scale of 1-5:
1 = Missing critical info, incoherent, or excessively verbose
2 = Covers some points but major gaps or poor structure  
3 = Adequate coverage, readable but could be improved
4 = Good coverage, well-structured, minor issues only
5 = Excellent: concise, comprehensive, well-organized, easy to read

## User Question
{input_q}

## Summary to Evaluate
{output[:3000]}

## Source Material
{ctx_text[:5000]}

Respond in this exact format:
SCORE: <1-5>
REASON: <brief explanation>"""
        
        resp = judge.generate(prompt)
        return _parse_score_reason(resp, binary=False)
    
    metrics.append(("summarization", eval_summarization))
    
    # ── 5. Plan Quality (G-Eval, Rubric 1-5) ──
    def eval_plan_quality(input_q, plan_text, **_):
        if not plan_text:
            return 1.0, "No research plan was generated."
        
        prompt = f"""You are an impartial judge evaluating RESEARCH PLAN QUALITY.

Rate the plan on a scale of 1-5:
1 = Random/irrelevant sub-questions, no logical structure
2 = Some relevance but major gaps or redundancy
3 = Adequate: covers main aspects but could be more thorough
4 = Good: well-structured, covers key angles with clear reasoning
5 = Excellent: comprehensive, logical progression, no redundancy

## User Question
{input_q}

## Research Plan (Sub-queries generated by Manager Agent)
{plan_text}

Respond in this exact format:
SCORE: <1-5>
REASON: <brief explanation>"""
        
        resp = judge.generate(prompt)
        return _parse_score_reason(resp, binary=False)
    
    metrics.append(("plan_quality", eval_plan_quality))
    
    return metrics


def _parse_score_reason(response: str, binary: bool = True) -> tuple:
    """Parse 'SCORE: X\nREASON: Y' format from judge response."""
    score = 0.0
    reason = response
    
    for line in response.split("\n"):
        line = line.strip()
        if line.upper().startswith("SCORE:"):
            try:
                raw = line.split(":", 1)[1].strip()
                score = float(raw)
                if binary:
                    score = 1.0 if score >= 0.5 else 0.0
                else:
                    score = max(1.0, min(5.0, score))  # Clamp 1-5
            except ValueError:
                pass
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    
    return score, reason


# ═══════════════════════════════════════════════════════════════════════════
# LANGFUSE LOGGING
# ═══════════════════════════════════════════════════════════════════════════

def log_to_langfuse(query: str, results: Dict[str, tuple], elapsed: float, cost: Dict[str, float] = None, judge_calls: int = 0, cache_hit: bool = False):
    """Send evaluation scores to Langfuse."""
    try:
        import langfuse
        lf_client = langfuse.get_client()
        
        output_data = {
            "elapsed_sec": round(elapsed, 2),
            "cache_hit": cache_hit,
            **{name: {"score": s, "reason": r} for name, (s, r) in results.items()}
        }
        
        if cost:
            output_data["cost"] = cost
        
        # Include judge usage info in output for visibility
        if judge_calls > 0:
            output_data["judge_usage"] = {
                "model": "deepseek-reasoner",
                "calls": judge_calls,
                "est_input_tokens": judge_calls * 3000,
                "est_output_tokens": judge_calls * 400,
            }
        
        with lf_client.start_as_current_span(
            name="agent-quality-evaluation",
            input={"user_query": query},
            output=output_data
        ) as span:
            # Log Agent Costs as Scores (for Dashboard Charting)
            # Log Specific Costs as Scores (Granular breakdown for Dashboard)
            if cost:
                # 1. Agent LLM Cost
                if "model_cost" in cost:
                    span.score(
                        name="cost_agent_llm",
                        value=cost["model_cost"],
                        comment="Agent LLM Cost (USD)"
                    )
                
                # 2. Tool Costs (Breakdown)
                tool_breakdown = cost.get("tool_breakdown", {})
                for tool_key, amount in tool_breakdown.items():
                    # Log even if amount is 0 (e.g. free tools like Arxiv)
                    span.score(
                        name=f"cost_tool_{tool_key}",
                        value=amount,
                        comment=f"Tool Cost: {tool_key} (USD)"
                    )
                
                # Log Total Tool Cost (Combined)
                span.score(
                    name="cost_tools_total",
                    value=cost.get("tool_cost", 0.0),
                    comment="Total Tools Cost (USD)"
                )

            # 3. Judge Cost (Calculated here)
            judge_cost = 0.0
            if judge_calls > 0:
                judge_cost = _estimate_judge_cost(judge_calls)
                span.score(
                    name="cost_eval_judge",
                    value=judge_cost,
                    comment="Evaluator (Judge) LLM Cost (USD)"
                )
            
            # 4. Grand Total System Cost
            # cost["total"] includes agent (model + tools) but NOT judge
            # So Grand Total = cost["total"] + judge_cost
            grand_total = cost.get("total", 0.0) + judge_cost
            span.score(
                name="cost_total_system",
                value=grand_total,
                comment="Grand Total Cost (Agent + Tools + Judge)"
            )
            
            # 5. Latency (Seconds)
            span.score(
                name="latency_agent_sec",
                value=elapsed,
                comment="Agent Execution Latency (seconds)"
            )

            # 6. Cache Hit Status (Operational Metric)
            span.score(
                name="is_cache_hit",
                value=1 if cache_hit else 0,
                comment="1=Hit, 0=Miss"
            )

            for name, (score, reason) in results.items():
                span.score(
                    name=name,
                    value=score,
                    comment=reason[:200]
                )
        
        lf_client.flush()
        print("\n   -> Logged to Langfuse")
        
    except ImportError:
        print("\n   [WARN] Langfuse not installed, skipping.")
    except Exception as e:
        print(f"\n   [WARN] Langfuse logging failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_single_query(query_config: dict, judge: DeepSeekJudge, metrics: list, query_idx: int, total: int) -> dict:
    """
    Run agent + evaluate for a single test query.
    Returns dict with query, results, data, and normalized average.
    """
    query = query_config["query"]
    expected_tools = query_config.get("expected_tools", EXPECTED_TOOLS)
    category = query_config.get("category", "deep_research")

    print(f"\n{'─' * 64}")
    print(f"  [{query_idx}/{total}] {category.upper()}")
    print(f"  Query: \"{query[:70]}...\"" if len(query) > 70 else f"  Query: \"{query}\"")
    print(f"{'─' * 64}")

    # Run agent
    data = run_agent_and_capture(query)
    cache_hit = data.get("cache_hit", False)
    cost = data.get("cost", {})
    
    status = "CACHE HIT" if cache_hit else f"{len(data['retrieval_context'])} docs"
    print(f"  Results: {len(data['actual_output'])} chars | {status} | tools={data['tools_called']} | {data['elapsed_sec']:.1f}s")
    print(f"  Cost: tool=${cost.get('tool_cost', 0):.4f} + model=${cost.get('model_cost', 0):.4f} = ${cost.get('total', 0):.4f}")

    # Evaluate
    results = {}
    judge_calls = 0
    for name, eval_fn in metrics:
        print(f"  Evaluating: {name}...", end=" ", flush=True)
        try:
            # Skip metrics that don't apply to this query type
            skip_reason = _should_skip_metric(name, category, cache_hit, data)
            if skip_reason:
                results[name] = (1.0, skip_reason)
                print("SKIPPED")
                continue

            # Pass per-query expected_tools for tool_correctness
            score, reason = eval_fn(
                input_q=query,
                output=data["actual_output"],
                context=data["retrieval_context"],
                tools_called=data["tools_called"],
                plan_text=data["plan_text"],
                expected_tools=expected_tools,
            )
            results[name] = (score, reason)
            judge_calls += 1
            print(f"SCORE = {score}")
        except Exception as e:
            print(f"ERROR: {e}")
            results[name] = (0.0, f"Evaluation failed: {str(e)}")

    # Add judge cost
    judge_cost = _estimate_judge_cost(judge_calls)
    total_cost = cost.get("total", 0) + judge_cost

    # Normalize
    BINARY_METRICS = {"faithfulness", "answer_relevancy", "tool_correctness"}
    RUBRIC_METRICS = {"summarization", "plan_quality"}
    normalized_scores = {}
    for name, (score, _) in results.items():
        if name in RUBRIC_METRICS:
            normalized_scores[name] = score / 5.0
        else:
            normalized_scores[name] = score
    avg = sum(normalized_scores.values()) / len(normalized_scores) if normalized_scores else 0

    print(f"  -> Normalized Average: {avg:.2f} | Judge cost: ${judge_cost:.4f} | Total: ${total_cost:.4f}")

    # Log to Langfuse
    log_to_langfuse(
        query, 
        results, 
        data["elapsed_sec"], 
        cost={"tool_cost": cost.get('tool_cost', 0), "tool_breakdown": cost.get('tool_breakdown', {}), "model_cost": cost.get('model_cost', 0), "total": total_cost},
        judge_calls=judge_calls,
        cache_hit=data["cache_hit"]
    )

    return {
        "query": query,
        "category": category,
        "results": results,
        "normalized_avg": avg,
        "elapsed_sec": data["elapsed_sec"],
        "cost": total_cost,
    }


def _should_skip_metric(name: str, category: str, cache_hit: bool, data: dict) -> str:
    """
    Determine if a metric should be skipped for this query.
    Returns skip reason (str) or empty string if not skipped.
    """
    # General chat: skip metrics that need context/plan
    if category == "general_chat":
        if name in ("faithfulness", "summarization", "plan_quality"):
            return f"Skipped: not applicable for {category}"
    
    # Cache hit: skip metrics that need retrieval context or tools
    if cache_hit:
        if name == "faithfulness":
            return "Skipped: cache hit (no fresh retrieval context to verify against)"
        if name == "tool_correctness":
            return "Skipped: cache hit (no tools called, answer served from cache)"
        if name == "plan_quality":
            return "Skipped: cache hit (no research plan generated)"
    
    return ""


def main():
    print("=" * 64)
    print("   Agent Quality Evaluation (DeepEval Strategy)")
    print("   Judge: deepseek-reasoner (V3.2 Thinking Mode)")
    print("   Metrics: Faithfulness, Relevancy, Tool Correctness,")
    print("            Summarization, Plan Quality")
    print(f"   Queries: {len(TEST_QUERIES)} test cases")
    print(f"   Langfuse Tracing: {'ON' if _langfuse_handler else 'OFF'}")
    print("=" * 64)

    # 1. Initialize Judge
    print("\n[1/2] Initializing DeepSeek Judge...")
    judge = DeepSeekJudge()
    metrics = build_metrics(judge)
    print(f"   -> {len(metrics)} metrics ready.")

    # 2. Run all queries
    print(f"\n[2/2] Evaluating {len(TEST_QUERIES)} queries...")
    all_results = []
    for i, qcfg in enumerate(TEST_QUERIES, 1):
        result = evaluate_single_query(qcfg, judge, metrics, i, len(TEST_QUERIES))
        all_results.append(result)

    # ── FINAL SUMMARY ──
    print("\n" + "=" * 64)
    print("   FINAL EVALUATION SUMMARY")
    print("=" * 64)

    total_avg = 0
    total_cost = 0
    for r in all_results:
        icon = "[OK]" if r["normalized_avg"] >= 0.7 else "[!!]"
        print(f"  {icon}  {r['normalized_avg']:.2f}  |  ${r.get('cost', 0):.4f}  |  {r['category']:<14}  |  {r['query'][:45]}...")
        total_avg += r["normalized_avg"]
        total_cost += r.get("cost", 0)

    overall = total_avg / len(all_results) if all_results else 0
    print(f"\n  Overall Average : {overall:.2f}  (0 = worst, 1 = best)")
    print(f"  Total Time     : {sum(r['elapsed_sec'] for r in all_results):.1f}s")
    print(f"  Total Cost     : ${total_cost:.4f} (~{total_cost * 35:.1f} THB)")
    print(f"  Langfuse       : {'Logged ✅' if _langfuse_handler else 'Not available'}")
    print("=" * 64)


if __name__ == "__main__":
    main()
