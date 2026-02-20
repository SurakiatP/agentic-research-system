from typing import Dict, Any, Literal, List
from pydantic import BaseModel, Field
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.settings import settings
from src.utils.logger import logger
from src.core.state import AgentState, ResearchResult

DEFAULT_USER_ID = "default_user"


# === Structured Output for Router ===

class RouteDecision(BaseModel):
    """Structured output for route classification."""
    decision: Literal["general_chat", "deep_research"] = Field(
        description="Route decision: 'general_chat' for simple queries, 'deep_research' for complex research queries"
    )
    reasoning: str = Field(
        description="Brief explanation of why this route was chosen"
    )


def _get_llm(temperature: float = 0.0) -> ChatDeepSeek:
    """Helper to create LLM instance with consistent config."""
    model_config = settings.model_config_yaml.get("default_model", {})
    return ChatDeepSeek(
        model=model_config.get("name", "deepseek-chat"),
        temperature=temperature,
        max_tokens=model_config.get("max_tokens", 8192),
        api_key=settings.DEEPSEEK_API_KEY,
        api_base=model_config.get("api_base", "https://api.deepseek.com")
    )


# ============================================================
# Node 1: Router — Classify intent (general_chat vs deep_research)
# ============================================================

def router_node(state: AgentState) -> Dict[str, Any]:
    """
    Router Node: Uses LLM + structured output to classify user intent.
    - general_chat: simple greetings, casual questions, no research needed
    - deep_research: complex queries requiring multi-source investigation
    """
    logger.info("--- (Node) Router Agent ---")
    user_query = state["user_query"]

    llm = _get_llm(temperature=0.0)
    system_prompt = settings.prompts.get("router_agent", {}).get(
        "system_prompt",
        "You are an intent classifier. Decide if a query needs deep research or is just a general chat."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "User Query: {query}\n\nClassify this query.")
    ])

    structured_llm = llm.with_structured_output(RouteDecision)
    chain = prompt | structured_llm

    try:
        result: RouteDecision = chain.invoke({"query": user_query})
        logger.info(f"Router decision: {result.decision} (reason: {result.reasoning})")
        return {"route_decision": result.decision}

    except Exception as e:
        logger.error(f"Router failed: {e}. Defaulting to deep_research.")
        return {"route_decision": "deep_research"}


# ============================================================
# Node 2: General Chat — Answer simple queries directly
# ============================================================

def general_chat_node(state: AgentState) -> Dict[str, Any]:
    """
    General Chat Node: Answers simple/casual queries with memory.
    - STM: Uses recent conversation history from state messages
    - LTM: Loads user facts for personalization, extracts new facts after answering
    """
    logger.info("--- (Node) General Chat ---")
    user_query = state["user_query"]

    # === 1. Load Memory Context ===
    memory_context = ""

    # LTM: Load user facts
    try:
        from src.memory.long_term_memory import long_term_memory
        user_facts = long_term_memory.get_user_facts(DEFAULT_USER_ID)
        if user_facts:
            facts_str = "\n".join([f"- {fact}" for fact in user_facts])
            memory_context += f"\n\nKnown facts about this user:\n{facts_str}"
            logger.info(f"Loaded {len(user_facts)} user facts from LTM")
    except Exception as e:
        logger.debug(f"LTM user facts load skipped: {e}")

    # STM: Use recent messages from state (last 10 turns)
    chat_history_msgs = []
    messages = state.get("messages", [])
    recent_messages = messages[-10:] if len(messages) > 10 else messages
    for msg in recent_messages:
        if hasattr(msg, 'type') and hasattr(msg, 'content'):
            role = "user" if msg.type == "human" else "assistant"
            chat_history_msgs.append((role, msg.content))

    if chat_history_msgs:
        memory_context += f"\n\nRecent conversation history ({len(chat_history_msgs)} messages available)."
        logger.info(f"Loaded {len(chat_history_msgs)} messages from STM")

    # === 2. Build Prompt ===
    llm = _get_llm(temperature=0.7)
    system_prompt = settings.prompts.get("router_agent", {}).get(
        "general_chat_prompt",
        "You are a friendly and helpful AI assistant. Answer the user's question naturally and concisely."
    )

    if memory_context:
        system_prompt += memory_context

    # Build message list with chat history for multi-turn context
    prompt_messages = [("system", system_prompt)]
    for role, content in chat_history_msgs:
        prompt_messages.append((role, content))
    prompt_messages.append(("human", "{query}"))

    prompt = ChatPromptTemplate.from_messages(prompt_messages)
    chain = prompt | llm | StrOutputParser()

    try:
        answer = chain.invoke({"query": user_query})
        logger.info(f"General chat answered: {answer[:100]}...")

        # === 3. Post-processing: Extract & Store User Facts (LTM) ===
        try:
            from src.memory.long_term_memory import long_term_memory
            _extract_and_store_user_facts(user_query, answer, long_term_memory)
        except Exception as e:
            logger.debug(f"User fact extraction skipped: {e}")

        return {
            "final_answer": answer,
            "general_chat_answer": answer
        }

    except Exception as e:
        logger.error(f"General chat failed: {e}")
        return {"final_answer": f"Sorry, I couldn't process your message: {str(e)}"}


def _extract_and_store_user_facts(user_query: str, answer: str, ltm):
    """
    Use LLM to extract key facts about the user from the conversation.
    e.g., "My name is John" → fact: "User's name is John"
    Only extracts if the user reveals personal info.
    """
    try:
        llm = _get_llm(temperature=0.0)
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract any personal facts the user revealed about themselves from this conversation.
Only extract FACTS about the user (name, job, interests, preferences, location, etc.).
If no personal facts are revealed, respond with exactly: NONE
If facts are found, list them one per line, e.g.:
- User's name is John
- User is interested in AI
Do NOT include facts about the assistant or general knowledge."""),
            ("human", f"User said: {user_query}\nAssistant answered: {answer}")
        ])
        chain = extraction_prompt | llm | StrOutputParser()
        result = chain.invoke({})

        if result.strip().upper() != "NONE" and result.strip():
            facts = [line.strip().lstrip("- ") for line in result.strip().split("\n") if line.strip()]
            for fact in facts[:5]:  # Max 5 facts per turn
                if len(fact) > 5:  # Ignore very short noise
                    ltm.save_user_fact(DEFAULT_USER_ID, fact)
            logger.info(f"Extracted {len(facts)} user facts")
    except Exception as e:
        logger.debug(f"Fact extraction failed: {e}")


# ============================================================
# Node 3: Summarize Answer — Synthesize research results
# (Replaces the old summarizer_agent.py)
# ============================================================

def summarize_answer_node(state: AgentState) -> Dict[str, Any]:
    """
    Summarize Answer Node: Synthesizes all research findings into a final answer.
    Uses LTM to load user's preferred answer style (from past rejection feedback).
    After producing the answer, the graph will interrupt for HITL #2 review.
    """
    logger.info("--- (Node) Summarize Answer ---")

    research_results = state.get("research_results", [])
    user_query = state.get("user_query")
    reflection_feedback = state.get("reflection_feedback")
    reflection_count = state.get("reflection_count", 0)

    if not research_results:
        return {"final_answer": "No research results were found to answer your query."}

    # 1. Prepare Context from research results
    context = ""
    for res in research_results:
        context += f"## Source: {res.source}\n"
        context += f"Query: {res.query}\n"
        context += f"Content: {res.content}\n\n"

    # 2. Initialize LLM
    llm = _get_llm(temperature=0.3)

    # 3. Prepare Prompt — include style preferences from LTM
    system_prompt = settings.prompts.get("router_agent", {}).get(
        "summarize_prompt",
        settings.prompts.get("summarizer_agent", {}).get(
            "system_prompt", "You are a Research Summarizer."
        )
    )

    # Load user style preferences from LTM
    style_context = ""
    try:
        from src.memory.long_term_memory import long_term_memory
        style_prefs = long_term_memory.get_style_preferences(DEFAULT_USER_ID)
        if style_prefs:
            prefs_str = "\n".join([f"- {pref}" for pref in style_prefs])
            style_context = f"\n\nIMPORTANT — The user has these known answer style preferences (learned from past feedback):\n{prefs_str}\nPlease follow these style preferences when writing your synthesis."
            logger.info(f"Loaded {len(style_prefs)} style preferences from LTM")
    except Exception as e:
        logger.debug(f"Style preferences load skipped: {e}")

    if style_context:
        system_prompt += style_context

    human_message = "User Query: {query}\n\nResearch Findings:\n{context}\n\nSynthesize a comprehensive answer based ONLY on the findings above."

    # If there's reflection feedback, include it
    if reflection_feedback and reflection_count > 0:
        human_message += f"\n\n⚠️ IMPORTANT — The user previously rejected the answer with this feedback:\n\"{reflection_feedback}\"\nPlease address this feedback specifically in your new synthesis."

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_message)
    ])

    # 4. Chain
    chain = prompt | llm | StrOutputParser()

    try:
        logger.info(f"Generating summary (reflection #{reflection_count})...")
        final_answer = chain.invoke({
            "query": user_query,
            "context": context
        })


        # Log to Long Term Memory
        try:
            from src.memory.long_term_memory import long_term_memory
            long_term_memory.log_research(user_query, final_answer)
        except Exception as ltm_err:
            logger.debug(f"LTM logging skipped: {ltm_err}")

        # Cache the final answer
        try:
            from src.cache.semantic_cache import semantic_cache
            semantic_cache.set(ResearchResult(
                query=user_query,
                content=final_answer,
                source="summarizer",
                type="final_answer",
                confidence=1.0
            ))
            logger.info(f"Cached final answer for query: '{user_query[:80]}'")
        except Exception as cache_err:
            logger.debug(f"Final answer caching skipped: {cache_err}")

        return {"final_answer": final_answer}

    except Exception as e:
        logger.error(f"Summarize answer failed: {e}")
        return {"final_answer": f"Error generating summary: {str(e)}"}
