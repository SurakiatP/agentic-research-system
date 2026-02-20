from src.core.state import ResearchResult
from src.cache.semantic_cache import semantic_cache
import time

print("=== Testing Cache Filtering (Granularity Check) ===")

query_text = "What is the future of AI agents?"
tool_content = "Tool finding: Agents are evolving..."
final_content = "Final Report: The future of AI agents is hybrid..."

# 1. Create Sub-Query Result (Tool)
res_tool = ResearchResult(
    query=query_text,
    content=tool_content,
    source="tool_test",
    type="tool",
    confidence=0.9
)

# 2. Create Final Answer Result (Final Answer)
res_final = ResearchResult(
    query=query_text,
    content=final_content,
    source="summarizer_test",
    type="final_answer",
    confidence=1.0
)

print(f"Setting Tool Result for: '{query_text}'")
semantic_cache.set(res_tool)
time.sleep(1)

print(f"Setting Final Answer Result for: '{query_text}'")
semantic_cache.set(res_final)
time.sleep(1)

# 3. Test Retrieval with Filter: TOOL
print("\n--- Testing Filter: TOOL ---")
tool_hits = semantic_cache.get(query_text, filter_type="tool")
if tool_hits:
    res = tool_hits[0]
    print(f"✅ Got Result Type: {res.type}")
    print(f"   Content: {res.content[:20]}...")
    if res.type == "tool" and res.content == tool_content:
        print("✅ Success: Retrieved correct Tool result")
    else:
        print(f"❌ Failed: Expected tool content, got {res.content}")
else:
    print("❌ Failed: No result found for filter='tool'")

# 4. Test Retrieval with Filter: FINAL_ANSWER
print("\n--- Testing Filter: FINAL_ANSWER ---")
final_hits = semantic_cache.get(query_text, filter_type="final_answer")
if final_hits:
    res = final_hits[0]
    print(f"✅ Got Result Type: {res.type}")
    print(f"   Content: {res.content[:20]}...")
    if res.type == "final_answer" and res.content == final_content:
        print("✅ Success: Retrieved correct Final Answer result")
    else:
        print(f"❌ Failed: Expected final answer, got {res.content}")
else:
    print("❌ Failed: No result found for filter='final_answer'")

# 5. Test Retrieval without Filter (Ambiguous)
print("\n--- Testing No Filter (Ambiguous) ---")
ambiguous_hits = semantic_cache.get(query_text)
if ambiguous_hits:
    res = ambiguous_hits[0]
    print(f"ℹ️  Got Result Type: {res.type}")
    print(f"   Content: {res.content[:20]}...")
else:
    print("❌ Failed: No result found")
