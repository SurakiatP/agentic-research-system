from fastmcp import FastMCP
import requests
from config.settings import settings
from src.utils.logger import logger

# Langfuse cost tracking (graceful fallback if SDK v3+ without decorators)
try:
    from langfuse.decorators import observe, langfuse_context
except ImportError:
    def observe(**kwargs):
        def wrapper(fn):
            return fn
        return wrapper

# Initialize FastMCP Server
mcp = FastMCP("Brave Search")

@mcp.tool()
def brave_web_search(query: str, max_results: int = 5) -> str:
    """
    Perform a web search using Brave Search API.
    """
    return run_brave_search(query, max_results)

@observe(as_type="generation")
def run_brave_search(query: str, max_results: int = 5, count: int = None) -> str:
    """
    Core logic for Brave search (callable directly).
    """
    # Handle alias: if 'count' is provided by LLM, use it as max_results
    if count is not None:
        max_results = count
    if not settings.BRAVE_API_KEY:
        return "Error: Brave API Key not configured."

    try:
        langfuse_context.update_current_observation(
            level="DEBUG",
            cost=settings.TOOL_COSTS.get("brave", 0.0)
        )
    except Exception:
        pass

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "X-Subscription-Token": settings.BRAVE_API_KEY,
        "Accept": "application/json"
    }
    params = {
        "q": query,
        "count": max_results
    }

    try:
        logger.info(f"Searching Brave for: {query}")
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        results = data.get("web", {}).get("results", [])
        
        if not results:
            return "No results found."

        formatted = ""
        for i, res in enumerate(results, 1):
            formatted += f"Source {i}: {res.get('title')}\n"
            formatted += f"URL: {res.get('url')}\n"
            formatted += f"Content: {res.get('description')}\n\n"
            
        return formatted

    except Exception as e:
        logger.error(f"Brave search failed: {e}")
        return f"Error executing Brave search: {str(e)}"

if __name__ == "__main__":
    mcp.run()
