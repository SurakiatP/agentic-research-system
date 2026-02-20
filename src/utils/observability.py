"""
Observability Utilities
=======================
Centralized logic for calculating costs, latency, and performance scores 
for both Test Scripts and Production App (Gradio).

Ensures consistency in dashboard metrics.
"""

from typing import List, Dict, Any, Optional
from config.settings import settings

# --- Constants ---
# Estimated average running time of a full Deep Research run (Miss)
# Updated to 120.0s as per user request.
ESTIMATED_AVG_TIME_PER_QUERY = 120.0

# Estimated average cost of a full Deep Research run (Miss)
ESTIMATED_AVG_COST_PER_QUERY = 0.05

# --- Tool Cost Map ---
# Maps tool names (from code) to config keys (in settings.TOOL_COSTS)
_TOOL_COST_MAP = {
    "tavily_search": "tavily",
    "brave_search": "brave",
    "crawl_url": "firecrawl",
    "arxiv_search": "arxiv"
}

def estimate_tool_cost(tools_called: List[str]) -> tuple:
    """
    Calculate tool costs based on usage.
    Returns: (total_cost, breakdown_dict)
    """
    total = 0.0
    breakdown = {}
    for tool_name in tools_called:
        # Resolve key
        key = _TOOL_COST_MAP.get(tool_name, tool_name)
        
        # Get cost from settings (default 0.0 if not found)
        cost = settings.TOOL_COSTS.get(key, 0.0)
        
        total += cost
        breakdown[key] = breakdown.get(key, 0.0) + cost
        
    return total, breakdown

def estimate_model_cost(input_tokens: int = 0, output_tokens: int = 0) -> float:
    """
    Estimate cost based on DeepSeek pricing (or default).
    Pricing: Input $0.55/1M, Output $2.19/1M (approx).
    """
    # Heuristic: 1 char ~= 0.25 tokens? Or just pass estimated tokens.
    # If we only have text, we estimate.
    
    # Fetch pricing from config/model_config.yaml (via settings)
    # Default to DeepSeek pricing if not found (Input $0.28, Output $0.42 as per user update)
    model_cfg = settings.model_config_yaml.get("default_model", {})
    
    # Defaults in case config is missing (using user's new values as fallback)
    price_input = model_cfg.get("input_price", 0.28)
    price_output = model_cfg.get("output_price", 0.42)
    
    cost = (input_tokens / 1_000_000 * price_input) + (output_tokens / 1_000_000 * price_output)
    return cost

def calculate_production_scores(
    elapsed_sec: float,
    cache_hit: bool,
    tools_called: List[str],
    model_cost: float = 0.0
) -> Dict[str, Any]:
    """
    Generate the standard set of scores for Langfuse Dashboard.
    """
    # 1. Tool Costs
    tool_cost_total, tool_breakdown = estimate_tool_cost(tools_called)
    
    scores = {}

    # --- Latency ---
    scores["latency_agent_sec"] = elapsed_sec

    # --- Cache Metrics ---
    scores["is_cache_hit"] = 1 if cache_hit else 0
    
    if cache_hit:
        # Savings
        scores["cost_savings_usd"] = ESTIMATED_AVG_COST_PER_QUERY
        scores["time_savings_sec"] = ESTIMATED_AVG_TIME_PER_QUERY
    else:
        scores["cost_savings_usd"] = 0.0
        scores["time_savings_sec"] = 0.0

    # --- Cost Breakdown ---
    scores["cost_agent_llm"] = model_cost
    scores["cost_tools_total"] = tool_cost_total
    
    # Eval Judge is 0 in production
    scores["cost_eval_judge"] = 0.0
    
    # Grand Total
    scores["cost_total_system"] = model_cost + tool_cost_total + 0.0 # judge

    # Add individual tool costs for granular charting
    for tool_key, amount in tool_breakdown.items():
        scores[f"cost_tool_{tool_key}"] = amount

    return scores

def log_scores_to_langfuse(trace_id: str, scores: Dict[str, Any]):
    """
    Log calculated scores to a specific Langfuse trace by ID.
    Uses the `create_score()` method from the installed SDK (v3+).
    """
    try:
        from langfuse import Langfuse
        
        # Initialize client (reads LANGFUSE_PUBLIC_KEY / SECRET_KEY from env)
        lf_client = Langfuse()
        
        print(f"📡 Logging {len(scores)} scores to Trace ID: {trace_id}")
        
        for name, val in scores.items():
            lf_client.create_score(
                trace_id=trace_id,
                name=name,
                value=float(val),
                data_type="NUMERIC",
            )
            
        lf_client.flush()
        
    except Exception as e:
        print(f"⚠️ Failed to log scores to Langfuse: {e}")
