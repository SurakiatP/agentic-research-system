from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate

from config.settings import settings
from src.utils.logger import logger
from src.core.state import AgentState, SubQuery

# Wrapper for structured output
class ResearchPlan(BaseModel):
    sub_queries: List[SubQuery] = Field(description="List of sub-queries to research")

def manager_node(state: AgentState) -> Dict[str, Any]:
    """
    Manager Agent Node:
    - Analyzes the user query.
    - Planning: Decomposes query into sub-queries.
    - Decision: Should we research more or answer?
    """
    logger.info("--- (Node) Manager Agent ---")
    user_query = state["user_query"]
    
    # 1. Initialize LLM
    # Use planner model (reasoner) if available, else default
    model_config = settings.model_config_yaml.get("default_model", {})
    llm = ChatDeepSeek(
        model=model_config.get("name", "deepseek-chat"),
        temperature=model_config.get("temperature", 0.0),
        max_tokens=model_config.get("max_tokens", 8192),
        api_key=settings.DEEPSEEK_API_KEY,
        api_base=model_config.get("api_base", "https://api.deepseek.com")
    )

    # 2. Prepare Prompt
    system_prompt = settings.prompts.get("manager_agent", {}).get("system_prompt", "You are a Research Manager.")
    
    # Check if this is a reflection loop (user rejected previous answer)
    reflection_feedback = state.get("reflection_feedback")
    reflection_count = state.get("reflection_count", 0)

    human_message = "User Query: {query}\n\nDecompose this into 3-5 distinct sub-queries for deep research."

    if reflection_feedback and reflection_count > 0:
        human_message += f"\n\n⚠️ IMPORTANT: The user previously rejected the research answer with this feedback:\n\"{reflection_feedback}\"\nPlease create NEW sub-queries that specifically address this feedback. Focus on areas the user felt were missing or inadequate."

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_message)
    ])

    # 3. Chain with Structured Output
    # DeepSeek supports JSON mode, but with_structured_output is cleaner if supported.
    # If not supported flawlessly, we might need a fallback. assuming it works.
    structured_llm = llm.with_structured_output(ResearchPlan)
    chain = prompt | structured_llm

    try:
        # 4. Invoke
        result: ResearchPlan = chain.invoke({"query": user_query})
        
        logger.info(f"Manager generated {len(result.sub_queries)} sub-queries.")
        for sq in result.sub_queries:
            logger.debug(f"- {sq.sub_query} ({sq.reasoning})")

        return {
            "sub_queries": result.sub_queries,
            "next_step": "researcher"
        }

    except Exception as e:
        logger.error(f"Manager Agent failed: {e}")
        # Fallback: Create one generic sub-query
        fallback_sq = SubQuery(
            original_query=user_query,
            sub_query=user_query,
            reasoning="Fallback due to planning failure.",
            status="pending"
        )
        return {
            "sub_queries": [fallback_sq],
            "next_step": "researcher"
        }
