import os
import sys
from dotenv import load_dotenv

# Add src to python path
sys.path.append(os.path.join(os.getcwd(), 'src'))

load_dotenv()

from src.mcp.servers.tavily_server import run_tavily_search
from src.mcp.servers.brave_server import run_brave_search
from src.mcp.servers.firecrawl_server import run_scrape_website
from src.mcp.servers.arxiv_server import run_arxiv_search

def test_tools():
    output_file = "test_tools_output_utf8.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        def log(msg):
            print(msg)
            f.write(str(msg) + "\n")

        log("--- Testing Tavily Search ---")
        try:
            tavily_result = run_tavily_search("current state of AI agents 2024")
            log(f"Tavily Output Type: {type(tavily_result)}")
            log(f"Tavily Output Preview: {str(tavily_result)[:500]}...")
        except Exception as e:
            log(f"Tavily Failed: {e}")

        log("\n--- Testing Brave Search ---")
        try:
            brave_result = run_brave_search("python programming language")
            log(f"Brave Output Type: {type(brave_result)}")
            log(f"Brave Output Preview: {str(brave_result)[:500]}...")
        except Exception as e:
            log(f"Brave Failed: {e}")

        log("\n--- Testing Firecrawl Scrape ---")
        try:
            # scraping a simple page, e.g., example.com
            firecrawl_result = run_scrape_website("https://example.com")
            log(f"Firecrawl Output Type: {type(firecrawl_result)}")
            log(f"Firecrawl Output Preview: {str(firecrawl_result)[:500]}...")
        except Exception as e:
            log(f"Firecrawl Failed: {e}")

        log("\n--- Testing arXiv Search ---")
        try:
            arxiv_result = run_arxiv_search("knowledge distillation")
            log(f"arXiv Output Type: {type(arxiv_result)}")
            log(f"arXiv Output Preview: {str(arxiv_result)[:500]}...")
        except Exception as e:
            log(f"arXiv Failed: {e}")

if __name__ == "__main__":
    test_tools()
