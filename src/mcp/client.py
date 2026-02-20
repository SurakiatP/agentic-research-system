from typing import List, Callable, Dict
from langchain_core.tools import tool, StructuredTool
from src.mcp.servers.tavily_server import run_tavily_search
from src.mcp.servers.brave_server import run_brave_search
from src.mcp.servers.firecrawl_server import run_scrape_website
from src.mcp.servers.arxiv_server import run_arxiv_search
from src.utils.logger import logger

class MCPClient:
    """
    Acts as a registry and dispatcher for MCP tools.
    In a full MCP architecture, this would connect to remote servers.
    For this monolithic agent, we import functions directly but keep the abstraction.
    """

    def __init__(self):
        self.tools: List[StructuredTool] = []
        self._register_tools()

    def _register_tools(self):
        """Register available tools."""
        
        # 1. Tavily Search
        self.tools.append(StructuredTool.from_function(
            func=run_tavily_search,
            name="tavily_search",
            description="Search the web using Tavily. Best for general knowledge."
        ))

        # 2. Brave Search
        self.tools.append(StructuredTool.from_function(
            func=run_brave_search,
            name="brave_search",
            description="Search the web using Brave. Use as backup or for privacy."
        ))

        # 3. Firecrawl Scraper
        self.tools.append(StructuredTool.from_function(
            func=run_scrape_website,
            name="scrape_website",
            description="Scrape a specific URL to get clean Markdown content."
        ))

        # 4. arXiv Search
        self.tools.append(StructuredTool.from_function(
            func=run_arxiv_search,
            name="arxiv_search",
            description="Search for academic papers on arXiv."
        ))

        logger.info(f"Registered {len(self.tools)} MCP tools.")

    def get_tools(self) -> List[StructuredTool]:
        """Return list of LangChain-compatible tools."""
        return self.tools

mcp_client = MCPClient()
