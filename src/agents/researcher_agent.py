from typing import Dict, Any, List
import json
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate

from config.settings import settings
from src.utils.logger import logger
from src.core.state import AgentState, ResearchResult, SubQuery
from src.cache.semantic_cache import semantic_cache
from src.mcp.client import mcp_client


def _react_research(query: str, tools: list, llm: ChatDeepSeek, system_prompt: str) -> tuple:
    """
    ReAct Pattern for a single sub-query:
    1. LLM thinks about which tool to use
    2. Execute the tool
    3. LLM observes the result and synthesizes an answer
    
    Supports multi-step reasoning if the first tool result is insufficient.
    """
    tool_map = {t.name: t for t in tools}
    tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in tools])

    react_prompt = f"""{system_prompt}

You have access to the following tools:
{tool_descriptions}

To use a tool, respond with EXACTLY this format:
TOOL: <tool_name>
JSON_ARGS: {{"param1": "value1", "param2": "value2"}}

Wait for the observation before providing your final ANSWER:.

Important:
- You MUST start your response with either "TOOL:" or "ANSWER:"
- For search tools (tavily_search, brave_search, arxiv_search, scrape_website), you can simply use 'QUERY: <your search string>' instead of JSON_ARGS.
- ALWAYS include the source URL for every fact or claim you use.
"""

    messages = [
        SystemMessage(content=react_prompt),
        HumanMessage(content=f"Research this sub-query thoroughly:\n{query}")
    ]

    max_steps = 3  # Max reasoning steps per sub-query
    observations = []
    tools_called = []

    for step in range(max_steps):
        try:
            response = llm.invoke(messages)
            response_text = response.content.strip()
            
            # Parse response
            if "ANSWER:" in response_text:
                answer = response_text.split("ANSWER:")[-1].strip()
                return answer, tools_called

            elif "TOOL:" in response_text:
                lines = response_text.split("\n")
                tool_name = None
                json_args = None
                query_str = None
                
                for line in lines:
                    line = line.strip()
                    if line.startswith("TOOL:"):
                        tool_name = line.replace("TOOL:", "").strip()
                    elif line.startswith("JSON_ARGS:"):
                        json_args = line.replace("JSON_ARGS:", "").strip()
                    elif line.startswith("QUERY:"):
                        query_str = line.replace("QUERY:", "").strip()

                if tool_name and tool_name in tool_map:
                    # 1. Parse Arguments (either JSON_ARGS or QUERY)
                    args = {}
                    if json_args:
                        try:
                            args = json.loads(json_args)
                        except Exception as e:
                            observation = f"[Error]: Invalid JSON_ARGS format — {str(e)}"
                            logger.error(f"ReAct failed to parse JSON_ARGS: {json_args}")
                            args = None
                    elif query_str:
                        # Map QUERY to the first parameter of the tool
                        args = {list(tool_map[tool_name].args.keys())[0]: query_str}

                    # 2. Execute Tool
                    if args is not None:
                        logger.info(f"ReAct Step {step+1}: Using {tool_name} with args {str(args)[:100]}...")
                        try:
                            result = tool_map[tool_name].func(**args)
                            observation = f"[{tool_name} result]: {result}"
                            tools_called.append(tool_name)
                        except Exception as e:
                            observation = f"[{tool_name} error]: {str(e)}"
                            logger.error(f"Tool {tool_name} execution failed: {e}")
                    else:
                        observation = "[Error]: Failed to parse tool arguments."
                else:
                    logger.warning(f"ReAct: Tool '{tool_name}' not found or parsing failed.")
                    observation = f"[Error]: Tool '{tool_name}' not available."

                observations.append(observation)
                messages.append(AIMessage(content=response_text))
                messages.append(HumanMessage(content=f"Observation:\n{observation}\n\nBased on this, provide your ANSWER: or use another TOOL:"))
            else:
                logger.debug("ReAct: LLM response didn't follow format, treating as answer")
                return response_text, tools_called

        except Exception as e:
            logger.error(f"ReAct step {step+1} failed: {e}")
            break

    # If we exhausted steps, combine observations into answer
    if observations:
        # Fallback: Synthesize an answer from the observations instead of returning raw logs
        logger.info("ReAct: Max steps reached. Synthesizing final answer from observations...")
        final_prompt = [
            SystemMessage(content="You are a helpful researcher. Synthesize a concise answer to the user's query based ONLY on the provided observations. You MUST include specific citations for every claim using [Title](URL) format."),
            HumanMessage(content=f"Query: {query}\n\nObservations:\n" + "\n".join(observations))
        ]
        try:
            final_response = llm.invoke(final_prompt)
            return final_response.content.strip(), tools_called
        except Exception as e:
            logger.error(f"Final synthesis failed: {e}")
            return "\n\n".join(observations), tools_called  # Ultimate fallback

    return f"Research on '{query}' did not yield results.", tools_called


def researcher_node(state: AgentState) -> Dict[str, Any]:
    """
    Researcher Agent Node (Architecture-aligned):
    - Processes ALL pending sub-queries (Parallel Pattern from diagram)
    - Each sub-query: Cache Check → if miss → ReAct tool execution
    - Returns all results at once to Summarizer
    """
    logger.info("--- (Node) Researcher Agent ---")
    sub_queries = state.get("sub_queries", [])
    research_results = state.get("research_results", [])

    # Get tools and LLM
    tools = mcp_client.get_tools()
    model_config = settings.model_config_yaml.get("default_model", {})
    llm = ChatDeepSeek(
        model=model_config.get("name", "deepseek-chat"),
        temperature=0.0,
        max_tokens=model_config.get("max_tokens", 8192),
        api_key=settings.DEEPSEEK_API_KEY,
        api_base=model_config.get("api_base", "https://api.deepseek.com")
    )
    system_prompt = settings.prompts.get("researcher_agent", {}).get(
        "system_prompt", "You are a meticulous Researcher Agent."
    )

    # Process ALL pending sub-queries (not just the first one)
    new_results = []
    tools_used = state.get("tools_called", [])
    for sq in sub_queries:
        if sq.status != "pending":
            continue

        query_text = sq.sub_query
        logger.info(f"Processing sub-query: {query_text}")

        # Per-sub-query Cache Check (matches diagram)
        # Filter: Only accept cached sub-query results (type='tool'), NOT final reports
        cached_results = semantic_cache.get(query_text, filter_type="tool")
        if cached_results:
            logger.info(f"Cache HIT for sub-query: {query_text}")
            sq.status = "completed"
            new_results.extend(cached_results)
            continue

        # Cache MISS -> ReAct Pattern
        logger.info(f"Cache MISS -> Running ReAct for: {query_text}")
        content, called = _react_research(query_text, tools, llm, system_prompt)

        # Track tools used
        tools_used.extend(called)

        # Create result
        new_result = ResearchResult(
            query=query_text,
            content=content[:5000],
            source="react_agent",
            type="tool",
            confidence=0.9
        )

        # Cache the result
        semantic_cache.set(new_result)

        # Mark completed
        sq.status = "completed"
        new_results.append(new_result)

    return {
        "sub_queries": sub_queries,
        "research_results": research_results + new_results,
        "tools_called": tools_used,
    }
