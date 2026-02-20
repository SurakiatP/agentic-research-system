import asyncio
from src.core.graph import graph_app
from src.core.state import AgentState
from src.utils.logger import logger

def main():
    print("🕵️ Deep Research Agent - CLI Mode")
    print("-----------------------------------")
    
    query = input("Enter research topic: ").strip()
    if not query:
        print("Exiting...")
        return

    print(f"\n🚀 Starting research on: {query}...\n")
    
    initial_state: AgentState = {
        "messages": [],
        "user_query": query,
        "sub_queries": [],
        "research_results": [],
        "final_answer": None,
        "next_step": None,
        "loop_count": 0
    }

    try:
        config = {"recursion_limit": 20}
        final_state = graph_app.invoke(initial_state, config=config)
        
        print("\n✅ Research Completed!")
        print("-----------------------------------")
        print(final_state.get("final_answer", "No answer generated."))
        
        # Optional: Print sources
        results = final_state.get("research_results", [])
        if results:
            print("\n📚 Sources:")
            for i, res in enumerate(results, 1):
                print(f"{i}. {res.source} ({res.type})")

    except Exception as e:
        logger.error(f"Process failed: {e}")
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
