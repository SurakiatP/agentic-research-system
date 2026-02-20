from fastmcp import FastMCP
from tavily import TavilyClient
from config.settings import settings
from src.utils.logger import logger

# Langfuse cost tracking (graceful fallback if SDK v3+ without decorators)
try:
    from langfuse.decorators import observe, langfuse_context
except ImportError:
    def observe(**kwargs):
        """No-op decorator fallback."""
        def wrapper(fn):
            return fn
        return wrapper

# Initialize FastMCP Server
mcp = FastMCP("Tavily Search")

# Initialize Client
tavily_client = None
if settings.TAVILY_API_KEY:
    try:
        tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)
        logger.info("Tavily Client initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize Tavily Client: {e}")

@mcp.tool()
def tavily_search(query: str, max_results: int = 5) -> str:
    """
    Perform a web search using Tavily.
    Best for general knowledge and recent events.
    """
    return run_tavily_search(query, max_results)

@observe(as_type="generation")
def run_tavily_search(query: str, max_results: int = 5) -> str:
    """
    Core logic for Tavily search (callable directly).
    """
    if not tavily_client:
        return "Error: Tavily API Key not configured."

    try:
        langfuse_context.update_current_observation(
            level="DEBUG",
            cost=settings.TOOL_COSTS.get("tavily", 0.0)
        )
    except Exception:
        pass

    try:
        logger.info(f"Searching Tavily for: {query}")
        response = tavily_client.search(
            query=query, 
            max_results=max_results,
            search_depth="advanced"
        )
        
        results = response.get("results", [])
        if not results:
            return "No results found."

        # Format output as a string (Markdown)
        formatted = ""
        for i, res in enumerate(results, 1):
            formatted += f"Source {i}: {res.get('title')}\n"
            formatted += f"URL: {res.get('url')}\n"
            formatted += f"Content: {res.get('content')}\n\n"
            
        return formatted

    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        return f"Error executing Tavily search: {str(e)}"

if __name__ == "__main__":
    mcp.run()
