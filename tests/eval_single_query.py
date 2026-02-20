"""
Small utility to run & evaluate a single query.
Usage:
    python -m tests.eval_single_query "Why is the sky blue?"
"""

import sys
import logging
import os

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests.eval_agent_quality import evaluate_single_query, build_metrics
from src.eval.llm import DeepSeekJudge

# Hide excessive logs
logging.getLogger("httpx").setLevel(logging.WARNING)

def main():
    # 1. Grab Query
    if len(sys.argv) > 1:
        # Join all args in case user didn't quote the query
        query = " ".join(sys.argv[1:])
    else:
        query = "What is the key feature of DeepSeek V3?"

    print(f"🔥 Starting Single Evaluation for query: \"{query}\"")

    # 2. Setup Judge
    judge = DeepSeekJudge()
    metrics = build_metrics(judge)

    # 3. Create Config
    # Default to expecting general search tools (tavily/brave)
    # The evaluation logic will adapt based on this.
    qcfg = {
        "query": query,
        "expected_tools": ["tavily_search"], 
        "category": "deep_research" 
    }

    # 4. Execute
    # This runs the agent, evaluates, logs to Langfuse, and prints results
    evaluate_single_query(qcfg, judge, metrics, 1, 1)

    print("\n✅ Done! Check Langfuse Dashboard.")

if __name__ == "__main__":
    main()
