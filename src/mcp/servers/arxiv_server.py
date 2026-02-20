from fastmcp import FastMCP
import arxiv
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
mcp = FastMCP("arXiv Search")

@mcp.tool()
def search_arxiv(query: str, max_results: int = 3) -> str:
    """
    Search for academic papers on arXiv.
    """
    return run_arxiv_search(query, max_results)

@observe(as_type="generation")
def run_arxiv_search(query: str, max_results: int = 3) -> str:
    """
    Core logic for arXiv search (callable directly).
    """
    try:
        langfuse_context.update_current_observation(
            level="DEBUG",
            cost=settings.TOOL_COSTS.get("arxiv", 0.0)
        )
    except Exception:
        pass

    try:
        logger.info(f"Searching arXiv for: {query}")
        
        # Construct client
        client = arxiv.Client()
        
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        formatted = ""
        for i, result in enumerate(client.results(search), 1):
            formatted += f"Paper {i}: {result.title}\n"
            formatted += f"Authors: {', '.join([a.name for a in result.authors])}\n"
            formatted += f"Published: {result.published.strftime('%Y-%m-%d')}\n"
            formatted += f"Summary: {result.summary}\n"
            formatted += f"PDF URL: {result.pdf_url}\n\n"
            
        if not formatted:
            return "No academic papers found."
            
        return formatted

    except Exception as e:
        logger.error(f"arXiv search failed: {e}")
        return f"Error executing arXiv search: {str(e)}"

if __name__ == "__main__":
    mcp.run()
