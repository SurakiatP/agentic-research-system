from langgraph.checkpoint.memory import MemorySaver

class ShortTermMemory:
    """
    Wrapper for LangGraph's MemorySaver.
    Provides in-memory checkpointer for the graph state during execution.
    """
    
    def __init__(self):
        self.checkpointer = MemorySaver()

    def get_checkpointer(self):
        return self.checkpointer

short_term_memory = ShortTermMemory()
