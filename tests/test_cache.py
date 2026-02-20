from src.core.state import ResearchResult
from src.cache.semantic_cache import semantic_cache
import time

print("=== Testing Standard Schema Cache (Reverted) ===")

# 1. Create a dummy result
test_query = "Who is the CEO of Tesla?"
test_content = "Elon Musk is the CEO of Tesla."
test_source = "test_script_reverted"

result = ResearchResult(
    query=test_query,
    content=test_content,
    source=test_source,
    type="test",
    confidence=1.0
)

# 2. Set Cache
print(f"Setting cache for: '{test_query}'")
semantic_cache.set(result)

# Wait a moment for Redis async write (if any)
time.sleep(1)

# 3. Get Cache
print(f"Getting cache for: '{test_query}'")
cached_results = semantic_cache.get(test_query)

if cached_results:
    res = cached_results[0]
    print(f"✅ Cache Hit!")
    print(f"   Query: {res.query}")
    print(f"   Content: {res.content}")
    print(f"   Source: {res.source}")
    
    # Check if we got the correct fields back (not packed JSON)
    if res.content == test_content and res.source == test_source:
        print("✅ Content & Source match perfectly (Standard Schema)")
    else:
        print(f"❌ Content mismatch!\nExpected: {test_content}\nGot: {res.content}")
else:
    print("❌ Cache Miss (Failed to retrieve)")

print("\n=== Testing RapidFuzz Fuzzy Match ===")
fuzzy_query = "who is ceo of tessla" # Typo
cached_fuzzy = semantic_cache.get(fuzzy_query)

if cached_fuzzy:
    print(f"✅ Fuzzy Match Hit! '{fuzzy_query}' -> '{cached_fuzzy[0].query}'")
else:
    print("❌ Fuzzy Match Failed")
