from typing import Literal, Dict, Any
from langgraph.graph import StateGraph, END
from src.core.state import AgentState
from src.agents.manager_agent import manager_node
from src.agents.researcher_agent import researcher_node
from src.agents.router_agent import router_node, general_chat_node, summarize_answer_node
from src.cache.semantic_cache import semantic_cache
from src.memory.short_term_memory import short_term_memory
from src.utils.logger import logger

MAX_REFLECTION_LOOPS = 3


def cache_check_node(state: AgentState) -> Dict[str, Any]:
    """
    Entry point node: Check Semantic Cache for the user's original query.
    If cache hit → skip everything and return cached answer.
    If cache miss → proceed to Router for intent classification.
    (No LLM call — purely cache lookup)
    """
    logger.info("--- (Node) Cache Check ---")
    user_query = state["user_query"]

    cached_answer = semantic_cache.check_user_query(user_query)
    if cached_answer:
        logger.info(f"Cache HIT for user query: '{user_query[:80]}...'")
        return {
            "cache_hit": True,
            "cached_answer": cached_answer,
            "final_answer": cached_answer
        }
    
    logger.info(f"Cache MISS for user query: '{user_query[:80]}...'")
    return {"cache_hit": False}


def route_cache_check(state: AgentState) -> Literal["router", END]:
    """
    Route after cache check:
    - Cache hit → END (answer already set)
    - Cache miss → Router for intent classification
    """
    if state.get("cache_hit"):
        return END
    return "router"


def route_router(state: AgentState) -> Literal["general_chat", "manager"]:
    """
    Route after Router's classification:
    - general_chat → General Chat node (simple answer)
    - deep_research → Manager for sub-query planning
    """
    decision = state.get("route_decision", "deep_research")
    if decision == "general_chat":
        return "general_chat"
    return "manager"


def route_manager(state: AgentState) -> Literal["researcher", END]:
    """
    Route after Manager's planning.
    Manager always routes to Researcher (HITL interrupt happens before Researcher).
    """
    next_step = state.get("next_step")
    if next_step == "researcher":
        return "researcher"
    return END


def route_summarize_answer(state: AgentState) -> Literal["manager", END]:
    """
    Route after HITL #2 review of the final answer:
    - user_approved_final = True → END
    - user_approved_final = False → loop back to Manager (with feedback)
    - reflection_count >= MAX → force END (prevent infinite loops)
    """
    reflection_count = state.get("reflection_count", 0)
    user_approved = state.get("user_approved_final")

    if reflection_count >= MAX_REFLECTION_LOOPS:
        logger.warning(f"Max reflection loops ({MAX_REFLECTION_LOOPS}) reached. Ending.")
        return END

    if user_approved is False:
        logger.info(f"User rejected final answer. Looping back to Manager (reflection #{reflection_count + 1}).")
        return "manager"

    return END


def create_graph():
    """
    Constructs the LangGraph workflow:
    
    cache_check → [HIT] → END
                → [MISS] → router → [general_chat] → general_chat → END
                                   → [deep_research] → manager → [HITL #1] → researcher
                                                                              → summarize_answer → [HITL #2]
                                                                                  → [approve] → END
                                                                                  → [reject] → manager (loop)
    """
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    workflow.add_node("cache_check", cache_check_node)
    workflow.add_node("router", router_node)
    workflow.add_node("general_chat", general_chat_node)
    workflow.add_node("manager", manager_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("summarize_answer", summarize_answer_node)

    # 2. Entry Point
    workflow.set_entry_point("cache_check")

    # 3. Cache Check → Hit=END / Miss=Router
    workflow.add_conditional_edges(
        "cache_check",
        route_cache_check,
        {
            "router": "router",
            END: END
        }
    )

    # 4. Router → General Chat / Manager
    workflow.add_conditional_edges(
        "router",
        route_router,
        {
            "general_chat": "general_chat",
            "manager": "manager"
        }
    )

    # 5. General Chat → END
    workflow.add_edge("general_chat", END)

    # 6. Manager → Researcher (HITL #1 interrupt before researcher)
    workflow.add_conditional_edges(
        "manager",
        route_manager,
        {
            "researcher": "researcher",
            END: END
        }
    )

    # 7. Researcher → Summarize Answer
    workflow.add_edge("researcher", "summarize_answer")

    # 8. Summarize Answer → [HITL #2] → END or loop back to Manager
    workflow.add_conditional_edges(
        "summarize_answer",
        route_summarize_answer,
        {
            "manager": "manager",
            END: END
        }
    )

    # 9. Compile with HITL interrupts at TWO points
    checkpointer = short_term_memory.get_checkpointer()
    app = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["researcher"],     # HITL #1: review sub-queries
        interrupt_after=["summarize_answer"] # HITL #2: review final answer
    )
    
    logger.info("LangGraph workflow compiled successfully.")
    return app

# Singleton instance
graph_app = create_graph()
