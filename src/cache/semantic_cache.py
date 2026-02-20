from typing import Optional, List, Dict, Any
import json
import time
from src.utils.logger import logger
from src.core.state import ResearchResult


class SemanticCache:
    """
    Semantic Cache layer using RedisVL + RapidFuzz (two-stage check).
    
    Stage 1 (Fast): RapidFuzz fuzzy matching against known queries (>90% = hit).
    Stage 2 (Deep): Embedding-based vector similarity search via RedisVL.
    
    Uses lazy initialization to avoid crashes when Redis is not available.
    """

    def __init__(self):
        self.index = None
        self.embedder = None
        self._initialized = False
        self._known_queries: Dict[str, str] = {}  # normalized_query -> redis_key

    def _lazy_init(self):
        """Initialize Redis and Embedding Model on first use."""
        if self._initialized:
            return
        self._initialized = True

        try:
            from redisvl.index import SearchIndex
            from redis import Redis
            from sentence_transformers import SentenceTransformer
            from config.settings import settings

            self.redis_url = settings.REDIS_URL
            redis_vl_config = settings.cache_config.get("redis_vl", {})
            self.index_name = redis_vl_config.get("index_name", "research_cache_v2")
            self.distance_threshold = redis_vl_config.get("distance_threshold", 0.3)
            self.ttl = redis_vl_config.get("ttl_seconds", 3600)
            overwrite_index = redis_vl_config.get("overwrite_index", False)

            # Initialize Embedding Model
            model_name = settings.model_config_yaml.get("embedding_model", {}).get("name", "Qwen/Qwen3-Embedding-0.6B")
            logger.info(f"Loading embedding model: {model_name}...")
            self.embedder = SentenceTransformer(model_name)

            # Initialize Redis Index (with result_type tag for Level 1/2 separation)
            schema = {
                "index": {
                    "name": self.index_name,
                    "prefix": "cache:research_agent",
                },
                "fields": [
                    {"name": "query", "type": "text"},
                    {"name": "content", "type": "text"},
                    {"name": "source", "type": "text"},
                    {"name": "result_type", "type": "tag"},
                    {"name": "embedding", "type": "vector", "attrs": {
                        "dims": 1024,
                        "algorithm": "flat",
                        "distance_metric": "cosine"
                    }}
                ]
            }

            self.index = SearchIndex.from_dict(schema)
            self.index.connect(self.redis_url)
            self.index.create(overwrite=overwrite_index)
            logger.info(f"Semantic Cache index '{self.index_name}' initialized (overwrite={overwrite_index}).")

            # Load existing keys into known_queries for RapidFuzz
            self._load_known_queries()

        except Exception as e:
            logger.warning(f"Semantic Cache unavailable (Redis may not be running): {e}")
            self.index = None

    def _load_known_queries(self):
        """Load existing cached queries from Redis for RapidFuzz matching."""
        if not self.index:
            return
        try:
            # Scan for existing cache keys
            cursor = 0
            while True:
                cursor, keys = self.index.client.scan(cursor, match="cache:research_agent:*", count=100)
                for key in keys:
                    key_str = key if isinstance(key, str) else key.decode("utf-8")
                    query_val = self.index.client.hget(key_str, "query")
                    if query_val:
                        query_str = query_val if isinstance(query_val, str) else query_val.decode("utf-8")
                        self._known_queries[query_str] = key_str
                if cursor == 0:
                    break
            if self._known_queries:
                logger.info(f"Loaded {len(self._known_queries)} known queries for RapidFuzz matching.")
        except Exception as e:
            logger.debug(f"Could not load known queries for RapidFuzz: {e}")

    def get(self, query: str, filter_type: Optional[str] = None) -> Optional[List[ResearchResult]]:
        """
        Two-stage cache retrieval.
        Args:
            query: The text to search for.
            filter_type: Optional 'result_type' to filter by (e.g., 'tool' or 'final_answer').
        """
        self._lazy_init()
        if not self.index or not self.embedder:
            return None

        # === Stage 1: RapidFuzz Fuzzy Match ===
        try:
            from src.cache.data_extractor import data_extractor

            # Only perform fuzzy match if no filter is applied (RapidFuzz doesn't support tag filtering directly on keys)
            # Or if we accept that fuzzy match might return something of wrong type, we should verify it after retrieval.
            
            existing_queries = list(self._known_queries.keys())
            if existing_queries:
                matched_query = data_extractor.find_similar_query(query, existing_queries, threshold=95.0)
                if matched_query:
                    redis_key = self._known_queries[matched_query]
                    data = self.index.client.hgetall(redis_key)
                    if data:
                        # Decode bytes (handle missing result_type for legacy data)
                        raw_type = data.get(b"result_type", data.get("result_type", b""))
                        r_type = raw_type.decode("utf-8") if isinstance(raw_type, bytes) else str(raw_type)
                        
                        # Apply Filter
                        if filter_type and r_type != filter_type:
                            logger.info(f"Cache SKIP (RapidFuzz): type mismatch for '{query}' — cached={r_type}, wanted={filter_type}")
                            # Don't return — fall through to Vector Search
                        else:
                            raw_content = data.get(b"content", data.get("content", b""))
                            raw_source = data.get(b"source", data.get("source", b""))
                            content = raw_content.decode("utf-8") if isinstance(raw_content, bytes) else str(raw_content)
                            source = raw_source.decode("utf-8") if isinstance(raw_source, bytes) else str(raw_source)

                            logger.info(f"Cache HIT (RapidFuzz) for query: '{query}' ~= '{matched_query}' [type={r_type}]")
                            return [ResearchResult(
                                query=matched_query,
                                content=content,
                                source=source,
                                type=r_type or "cache_fuzzy",
                                confidence=0.95
                            )]
        except Exception as e:
            logger.debug(f"RapidFuzz cache check failed (non-critical): {e}")

        # === Stage 2: Embedding Vector Search ===
        try:
            from redisvl.query import VectorQuery
            from redisvl.query.filter import Tag

            query_vector = self.embedder.encode(query).tolist()
            
            # Build Filter Expression
            filter_expression = None
            if filter_type:
                filter_expression = Tag("result_type") == filter_type

            v_query = VectorQuery(
                vector=query_vector,
                vector_field_name="embedding",
                return_fields=["query", "content", "source", "result_type"],
                num_results=1,
                filter_expression=filter_expression
            )

            results = self.index.query(v_query)

            if results and len(results) > 0:
                match = results[0]
                distance = float(match.get("vector_distance", 1.0))
                matched_type = match.get('result_type', 'unknown')

                if distance <= self.distance_threshold:
                    logger.info(f"Cache HIT (Vector) for query: '{query}' (Distance: {distance:.4f}, type={matched_type})")
                    return [ResearchResult(
                        query=match['query'],
                        content=match['content'],
                        source=match['source'],
                        type=matched_type,
                        confidence=1.0 - distance
                    )]

            logger.debug(f"Cache MISS for query: '{query}'")
            return None

        except Exception as e:
            logger.error(f"Error reading from Semantic Cache: {e}")
            return None

    def set(self, result: ResearchResult):
        """Cache a new research result."""
        self._lazy_init()
        if not self.index or not self.embedder:
            return

        try:
            # Encode query and convert vector to bytes for Redis HASH storage
            import numpy as np
            vector = self.embedder.encode(result.query)
            vector_bytes = np.array(vector, dtype=np.float32).tobytes()

            data = {
                "query": result.query,
                "content": result.content,
                "source": result.source,
                "result_type": result.type,  # Store the type (tool/final_answer)
                "embedding": vector_bytes
            }

            # Use redisvl's load method which handles vector serialization
            keys = self.index.load([data])
            
            # Set TTL for the new keys
            if keys:
                for key in keys:
                    self.index.client.expire(key, self.ttl)
                    # Track for RapidFuzz
                    self._known_queries[result.query] = key

            logger.debug(f"Cached result for: '{result.query}' (Type: {result.type})")

        except Exception as e:
            logger.error(f"Error writing to Semantic Cache: {e}")

    def check_user_query(self, query: str) -> Optional[str]:
        """
        Check if the user's original query has a cached answer.
        Returns the cached TEXT content string if found.
        Strictly looks for 'final_answer' type.
        """
        results = self.get(query, filter_type="final_answer")
        if results:
            return results[0].content
        return None


semantic_cache = SemanticCache()
