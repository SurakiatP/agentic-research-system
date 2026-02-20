import json
import psycopg2
from psycopg2.extras import Json
from typing import Optional, Dict, Any, List
from src.utils.logger import logger

class LongTermMemory:
    """
    Long-Term Memory using PostgreSQL.
    Stores user preferences and research history for retrieval across sessions.
    Uses lazy initialization to avoid crashes when PostgreSQL is not available.
    """

    def __init__(self):
        self.conn = None
        self._initialized = False

    def _lazy_init(self):
        """Initialize PostgreSQL connection on first use."""
        if self._initialized:
            return
        self._initialized = True

        try:
            from config.settings import settings

            self.host = settings.PG_HOST
            self.port = settings.PG_PORT
            self.database = settings.PG_DATABASE
            self.user = settings.PG_USER
            self.password = settings.PG_PASSWORD

            if not self.password:
                logger.warning("PostgreSQL password not set. Long-Term Memory disabled.")
                return

            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            self.conn.autocommit = True
            logger.info(f"Connected to PostgreSQL database: {self.database}")
            self._init_db()

        except Exception as e:
            logger.warning(f"Long-Term Memory unavailable (PostgreSQL may not be running): {e}")
            self.conn = None

    def _init_db(self):
        """Create necessary tables if they don't exist."""
        if not self.conn:
            return

        queries = [
            """
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY,
                preferences JSONB DEFAULT '{}'::jsonb
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS research_logs (
                id SERIAL PRIMARY KEY,
                query TEXT NOT NULL,
                summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS kv_store (
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                value JSONB NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (namespace, key)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS conversation_logs (
                id SERIAL PRIMARY KEY,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        ]

        try:
            with self.conn.cursor() as cur:
                for query in queries:
                    cur.execute(query)
            logger.info("Database tables verified/created.")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")

    def save_preference(self, user_id: str, key: str, value: Any):
        """Save a specific preference for a user."""
        self._lazy_init()
        if not self.conn:
            return

        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO user_preferences (user_id, preferences)
                    VALUES (%s, %s)
                    ON CONFLICT (user_id) 
                    DO UPDATE SET preferences = user_preferences.preferences || %s;
                    """,
                    (user_id, Json({key: value}), Json({key: value}))
                )
            logger.debug(f"Saved preference for user {user_id}: {key}={value}")
        except Exception as e:
            logger.error(f"Failed to save preference: {e}")

    def get_preferences(self, user_id: str) -> Dict[str, Any]:
        """Retrieve all preferences for a user."""
        self._lazy_init()
        if not self.conn:
            return {}

        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT preferences FROM user_preferences WHERE user_id = %s;", (user_id,))
                result = cur.fetchone()
                if result:
                    return result[0]
        except Exception as e:
            logger.error(f"Failed to retrieve preferences: {e}")

        return {}

    def log_research(self, query: str, summary: str):
        """Log a completed research task."""
        self._lazy_init()
        if not self.conn:
            return

        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO research_logs (query, summary) VALUES (%s, %s)",
                    (query, summary)
                )
            logger.info(f"Logged research for query: {query}")
        except Exception as e:
            logger.error(f"Failed to log research: {e}")

    def save_kv(self, namespace: str, key: str, value: Any):
        """Save a key-value pair to the KV store."""
        self._lazy_init()
        if not self.conn:
            return

        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO kv_store (namespace, key, value, updated_at)
                    VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (namespace, key)
                    DO UPDATE SET value = %s, updated_at = CURRENT_TIMESTAMP;
                    """,
                    (namespace, key, Json(value), Json(value))
                )
            logger.debug(f"KV saved: {namespace}/{key}")
        except Exception as e:
            logger.error(f"Failed to save KV: {e}")

    def get_kv(self, namespace: str, key: str) -> Optional[Any]:
        """Retrieve a value from the KV store."""
        self._lazy_init()
        if not self.conn:
            return None

        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "SELECT value FROM kv_store WHERE namespace = %s AND key = %s;",
                    (namespace, key)
                )
                result = cur.fetchone()
                if result:
                    return result[0]
        except Exception as e:
            logger.error(f"Failed to get KV: {e}")
        return None

    def get_relevant_research(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search research_logs for past research relevant to the query (simple keyword match)."""
        self._lazy_init()
        if not self.conn:
            return []

        try:
            with self.conn.cursor() as cur:
                # Simple keyword search using ILIKE for relevance
                search_terms = query.split()[:5]  # Use first 5 words
                conditions = " OR ".join(["query ILIKE %s" for _ in search_terms])
                params = [f"%{term}%" for term in search_terms]
                params.append(limit)
                
                cur.execute(
                    f"SELECT query, summary, created_at FROM research_logs WHERE {conditions} ORDER BY created_at DESC LIMIT %s;",
                    params
                )
                rows = cur.fetchall()
                return [
                    {"query": r[0], "summary": r[1], "created_at": str(r[2])}
                    for r in rows
                ]
        except Exception as e:
            logger.error(f"Failed to search research logs: {e}")
        return []

    # === Conversation History (STM persistence for General Chat) ===

    def log_conversation(self, thread_id: str, role: str, content: str):
        """Log a single conversation turn (user or assistant)."""
        self._lazy_init()
        if not self.conn:
            return

        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO conversation_logs (thread_id, role, content) VALUES (%s, %s, %s)",
                    (thread_id, role, content)
                )
            logger.debug(f"Conversation logged: [{role}] {content[:60]}...")
        except Exception as e:
            logger.error(f"Failed to log conversation: {e}")

    def get_chat_history(self, thread_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """Get recent conversation history for a thread."""
        self._lazy_init()
        if not self.conn:
            return []

        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """SELECT role, content FROM conversation_logs 
                    WHERE thread_id = %s ORDER BY created_at DESC LIMIT %s;""",
                    (thread_id, limit)
                )
                rows = cur.fetchall()
                # Reverse to get chronological order
                return [{"role": r[0], "content": r[1]} for r in reversed(rows)]
        except Exception as e:
            logger.error(f"Failed to get chat history: {e}")
        return []

    # === Style Preferences (LTM for Summarize Answer) ===

    def save_style_preference(self, user_id: str, preference: str):
        """Save/append a style preference extracted from HITL #2 rejection feedback."""
        self._lazy_init()
        if not self.conn:
            return

        try:
            # Get existing preferences
            existing = self.get_kv("style_preferences", user_id) or []
            if isinstance(existing, list):
                # Avoid duplicates (case-insensitive)
                if preference.lower() not in [p.lower() for p in existing]:
                    existing.append(preference)
            else:
                existing = [preference]

            self.save_kv("style_preferences", user_id, existing)
            logger.info(f"Style preference saved for {user_id}: {preference}")
        except Exception as e:
            logger.error(f"Failed to save style preference: {e}")

    def get_style_preferences(self, user_id: str) -> List[str]:
        """Get all style preferences for a user."""
        result = self.get_kv("style_preferences", user_id)
        if isinstance(result, list):
            return result
        return []

    # === User Facts (LTM for General Chat) ===

    def save_user_fact(self, user_id: str, fact: str):
        """Save a fact about the user (e.g., name, interests) for cross-session memory."""
        self._lazy_init()
        if not self.conn:
            return

        try:
            existing = self.get_kv("user_facts", user_id) or []
            if isinstance(existing, list):
                if fact.lower() not in [f.lower() for f in existing]:
                    existing.append(fact)
                    # Keep max 20 facts
                    existing = existing[-20:]
            else:
                existing = [fact]

            self.save_kv("user_facts", user_id, existing)
            logger.info(f"User fact saved for {user_id}: {fact}")
        except Exception as e:
            logger.error(f"Failed to save user fact: {e}")

    def get_user_facts(self, user_id: str) -> List[str]:
        """Get all known facts about a user."""
        result = self.get_kv("user_facts", user_id)
        if isinstance(result, list):
            return result
        return []

    def close(self):
        if self.conn:
            self.conn.close()

long_term_memory = LongTermMemory()
