from fastmcp import FastMCP
from firecrawl import FirecrawlApp
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
mcp = FastMCP("Firecrawl Scraper")

# Initialize Client
firecrawl_app = None
if settings.FIRECRAWL_API_KEY:
    try:
        firecrawl_app = FirecrawlApp(api_key=settings.FIRECRAWL_API_KEY)
        logger.info("Firecrawl Client initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize Firecrawl: {e}")

@mcp.tool()
def scrape_website(url: str) -> str:
    """
    Scrape a website and convert it to clean Markdown.
    Best for deep reading of a specific URL.
    """
    return run_scrape_website(url)

@observe(as_type="generation")
def run_scrape_website(url: str) -> str:
    """
    Core logic for Firecrawl scraping (callable directly).
    """
    if not firecrawl_app:
        return "Error: Firecrawl API Key not configured."

    try:
        langfuse_context.update_current_observation(
            level="DEBUG",
            cost=settings.TOOL_COSTS.get("firecrawl", 0.0)
        )
    except Exception:
        pass

    try:
        logger.info(f"Scraping URL with Firecrawl: {url}")
        scrape_result = firecrawl_app.scrape(url, formats=['markdown'])
        
        # Handle Pydantic model or Dict return
        markdown_content = None
        if hasattr(scrape_result, 'markdown'):
            markdown_content = scrape_result.markdown
        elif isinstance(scrape_result, dict) and 'markdown' in scrape_result:
            markdown_content = scrape_result['markdown']
        
        if markdown_content:
            return markdown_content
        else:
            logger.error(f"Firecrawl output missing 'markdown'. Result: {scrape_result}")
            return "Error: No markdown content returned."

    except Exception as e:
        logger.error(f"Firecrawl scraping failed: {e}")
        return f"Error executing Firecrawl scraping: {str(e)}"

if __name__ == "__main__":
    mcp.run()
