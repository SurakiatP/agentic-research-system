import os
from dotenv import load_dotenv

# Load environment variables before doing anything else
load_dotenv()

from redis import Redis
from config.settings import settings
from src.utils.logger import logger
def clear_cache():
    try:
        redis_url = settings.REDIS_URL
        client = Redis.from_url(redis_url)
        
        # Option 1: Flush everything (Simplest)
        # client.flushdb()
        
        # Option 2: Delete only agent keys (Safer)
        prefix = "cache:research_agent:*"
        keys = []
        cursor = 0
        while True:
            cursor, batch = client.scan(cursor, match=prefix, count=100)
            keys.extend(batch)
            if cursor == 0:
                break
        
        if keys:
            client.delete(*keys)
            print(f"✅ Cleared {len(keys)} keys with prefix '{prefix}' from Redis.")
        else:
            print(f"ℹ️ No keys found with prefix '{prefix}'. Cache is already empty.")

        # Also drop the index to ensure schema changes if any (though not strictly needed for just data clearing)
        # But user wants to "clear cache". Index persistence is fine.
            
    except Exception as e:
        print(f"❌ Error clearing cache: {e}")

if __name__ == "__main__":
    clear_cache()
