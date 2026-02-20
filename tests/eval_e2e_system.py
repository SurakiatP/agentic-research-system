"""
End-to-End (E2E) System Evaluation Script
=========================================
Measures the REAL-WORLD performance savings of the Deep Research Agent.

Methodology:
1.  Cold Run (Cache Miss):
    -   Executes a complex query that forces full agent reasoning.
    -   Simulates user approval (Auto-Resume) at HITL breakpoints.
    -   Measures total execution time (Router -> Manager -> Researcher -> Summarizer).

2.  Warm Run (Cache Hit):
    -   Executes the SAME query immediately after.
    -   Should hit the Semantic Cache and return instantly.
    -   Measures total execution time.

3.  Result:
    -   Calculates "Real Savings" = Time(Cold) - Time(Warm).
    -   Logs the experiment to Langfuse.

Usage:
    python -m tests.eval_e2e_system
"""

import time
import os
import sys
import uuid
from typing import Dict, Any

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from src.core.graph import graph_app
from src.utils.logger import logger

# Disable detailed agent logs for cleaner output (optional)
# logger.setLevel("WARNING")

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

TEST_QUERY = "What are the latest breakthroughs in Quantum Machine Learning (QML) as of 2024-2025? specifically about error mitigation."
THREAD_ID = f"e2e_test_{uuid.uuid4().hex[:8]}"
CONFIG = {"configurable": {"thread_id": THREAD_ID}}


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def run_agent_with_auto_approve(query: str, thread_config: Dict[str, Any]) -> float:
    """
    Runs the agent for a given query, automatically approving all interrupts.
    Returns the total execution time in seconds.
    """
    print(f"   ▶ Starting run for query: '{query[:50]}...'")
    start_time = time.perf_counter()
    
    # 1. Initial invocation
    current_input = {"user_query": query}
    steps_count = 0
    
    # Use stream or simple invocation loop to handle interrupts
    # Logic: invoke -> if interrupt -> resume -> repeat until done
    
    while True:
        try:
            # We use invoke. If it hits an interrupt, it might raise GraphInterrupt 
            # or simply return the state depending on LangGraph version.
            # In latest versions, it stops. We check state.
            
            # Note: graph_app.invoke() typically runs until interrupt.
            # If interrupted, it saves state and stops.
            # We need to detect if it stopped due to interrupt or completion.
            
            result = graph_app.invoke(current_input, config=thread_config)
            
            # If result contains 'final_answer' and we are at END, we are done.
            # But invoke() returns the final state of that *run segment*.
            
            # Let's check the next step (interrupt)
            snapshot = graph_app.get_state(thread_config)
            next_node = snapshot.next
            
            if not next_node:
                # Execution finished completely
                print("     ✅ Execution Finished (End of Graph)")
                break
            
            print(f"     ⏸️  Interrupted at: {next_node}")
            
            # HANDLE INTERRUPTS
            if "researcher" in next_node:
                print("     👉 Auto-Approving Research Plan (HITL #1)...")
                # Resume (Proceed to Researcher)
                # Manager has presumably set 'sub_queries'. We just say "go ahead".
                current_input = None 
                
            elif "summarize_answer" in next_node:
                 # Note: interrupt is AFTER summarize_answer, so next might be empty or loop?
                 # Wait, graph definition: interrupt_after=["summarize_answer"]
                 # So it stops AFTER summarize_answer logic ran.
                 # The 'summarize_answer' node outputs 'final_answer'.
                 # The user needs to approve.
                 
                 print("     👉 Auto-Approving Final Answer (HITL #2)...")
                 # We must update state to say 'user_approved_final=True'
                 graph_app.update_state(thread_config, {"user_approved_final": True})
                 current_input = None
                 
            else:
                 # Fallback for unexpected interrupts
                 print(f"     👉 Resuming unexpected interrupt at {next_node}...")
                 current_input = None

            steps_count += 1
            if steps_count > 10:
                print("     ⚠️  Too many interrupts, aborting loop.")
                break
                
        except Exception as e:
            # Some versions raise GraphInterrupt.
            # If not, check if it's a real error.
            print(f"     ❌ Error during execution: {e}")
            break

    elapsed = time.perf_counter() - start_time
    return elapsed


def log_to_langfuse(cold_time: float, warm_time: float, query: str):
    """Log the E2E result to Langfuse."""
    try:
        import langfuse
        lf_client = langfuse.get_client()
        
        savings = cold_time - warm_time
        
        with lf_client.start_as_current_span(
            name="deep-research-e2e-test",
            input={"user_query": query},
            output={
                "cold_time_sec": round(cold_time, 2), 
                "warm_time_sec": round(warm_time, 2),
                "savings_sec": round(savings, 2)
            }
        ) as span:
            span.score(
                name="real_savings_sec",
                value=round(savings, 2),
                comment="Difference between Cold Run and Warm Run"
            )
            span.score(
                name="cold_run_sec",
                value=round(cold_time, 2)
            )
            span.score(
                name="warm_run_sec",
                value=round(warm_time, 2)
            )
            
        lf_client.flush()
        print("\n📡  Logged E2E results to Langfuse ✅")
        
    except ImportError:
        print("\n⚠️  Langfuse not installed.")
    except Exception as e:
        print(f"\n⚠️  Langfuse logging failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   🚀 End-to-End System Evaluation (Real-World Test)        ║")
    print("║   Goal: Measure ROI (Time Savings) of Semantic Cache       ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    
    print(f"\n📝 Test Query: \"{TEST_QUERY}\"")
    print(f"🆔 Thread ID: {THREAD_ID}")
    
    # --- PHASE 1: COLD RUN ---
    print("\n🥶 PHASE 1: Cold Run (Cache Miss)...")
    print("   (This will take time - running full agent research...)")
    
    cold_time = run_agent_with_auto_approve(TEST_QUERY, CONFIG)
    print(f"   ⏱️  Cold Run Time: {cold_time:.2f} seconds")
    
    # --- PHASE 2: WARM RUN ---
    print("\n🔥 PHASE 2: Warm Run (Cache Hit)...")
    
    # Use SAME thread ID? 
    # Logic: Semantic Cache works across threads (global Redis).
    # Thread ID only matters for LangGraph state checkpointer.
    # To be "User B asking same question", we can use a new thread BUT same query.
    # Actually, let's use the SAME thread to simulate "User asks again", 
    # OR new thread to simulate "Another user asks".
    # Cache is global, so NEW thread is a better proof of global cache.
    
    NEW_THREAD_ID = f"e2e_test_{uuid.uuid4().hex[:8]}"
    WARM_CONFIG = {"configurable": {"thread_id": NEW_THREAD_ID}}
    
    warm_time = run_agent_with_auto_approve(TEST_QUERY, WARM_CONFIG)
    print(f"   ⏱️  Warm Run Time: {warm_time:.2f} seconds")
    
    # --- REPORT ---
    savings = cold_time - warm_time
    savings_pct = (savings / cold_time * 100) if cold_time > 0 else 0
    
    print("\n" + "="*60)
    print("📊 E2E EVALUATION RESULTS")
    print("="*60)
    print(f"  Cold Run (Full Agent) : {cold_time:.2f} s")
    print(f"  Warm Run (Cache Hit)  : {warm_time:.2f} s")
    print("-" * 60)
    print(f"  💰 REAL TIME SAVINGS  : {savings:.2f} s")
    print(f"  📈 Efficiency Gain    : {savings_pct:.1f} %")
    print("="*60)
    
    if savings < 1.0:
        print("\n⚠️  WARNING: Savings are low. Did the Cold Run actually cache the result?")
        print("   Possible reasons:")
        print("   1. Summarizer didn't produce 'final_answer' type?")
        print("   2. Cache 'set' operation failed?")
        print("   3. Query was too simple (Cold run was fast)?")
    
    # --- LOG ---
    log_to_langfuse(cold_time, warm_time, TEST_QUERY)

if __name__ == "__main__":
    main()
