import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

def load_yaml_config(file_path: Path) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    if not file_path.exists():
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

class Settings(BaseSettings):
    """
    Centralized application configuration.
    Loads from .env file and environment variables.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_file_find_dotenv=True
    )

    # Project Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    CONFIG_DIR: Path = BASE_DIR / "config"
    DATA_DIR: Path = BASE_DIR / "data"
    LOG_DIR: Path = DATA_DIR / "logs"
    EXPORT_DIR: Path = DATA_DIR / "exports"

    # LLM Configuration
    DEEPSEEK_API_KEY: Optional[str] = Field(default=None, description="DeepSeek API Key")

    # Redis Configuration (Semantic Cache)
    REDIS_URL: str = Field(default="redis://localhost:6379", description="Redis connection URL for semantic cache")

    # PostgreSQL Configuration (Long Term Memory)
    PG_HOST: str = Field(default="localhost", description="PostgreSQL host")
    PG_PORT: int = Field(default=5432, description="PostgreSQL port")
    PG_DATABASE: str = Field(default="mcp_memory", description="PostgreSQL database name")
    PG_USER: str = Field(default="mcp_user", description="PostgreSQL username")
    PG_PASSWORD: Optional[str] = Field(default=None, description="PostgreSQL password")

    # MCP Tool Keys (Optional but recommended)
    TAVILY_API_KEY: Optional[str] = Field(default=None, description="Tavily API Key")
    BRAVE_API_KEY: Optional[str] = Field(default=None, description="Brave Search API Key")
    FIRECRAWL_API_KEY: Optional[str] = Field(default=None, description="Firecrawl API Key")

    # Observability (Optional)
    LANGFUSE_PUBLIC_KEY: Optional[str] = Field(default=None, description="Langfuse Public Key")
    LANGFUSE_SECRET_KEY: Optional[str] = Field(default=None, description="Langfuse Secret Key")
    LANGFUSE_BASE_URL: Optional[str] = Field(default=None, description="Langfuse Base URL (alias)")
    LANGFUSE_HOST: str = Field(default="https://cloud.langfuse.com", description="Langfuse Host URL")

    # Tool Costs (Estimated USD per call/page)
    TOOL_COSTS: Dict[str, float] = Field(
        default={
            "tavily": 0.008,
            "brave": 0.005,
            "firecrawl": 0.001,
            "arxiv": 0.0
        },
        description="Estimated cost per tool call"
    )

    # Dynamic Configs (Loaded from YAML)
    agent_config: Dict[str, Any] = {}
    model_config_yaml: Dict[str, Any] = {}
    mcp_config: Dict[str, Any] = {}
    cache_config: Dict[str, Any] = {}
    prompts: Dict[str, Any] = {}
    logging_config: Dict[str, Any] = {}

    def __init__(self, **values):
        super().__init__(**values)
        self._load_yaml_configs()
        self._ensure_dirs()

    def _load_yaml_configs(self):
        """Load additional configurations from YAML files."""
        self.agent_config = load_yaml_config(self.CONFIG_DIR / "agent_config.yaml")
        self.model_config_yaml = load_yaml_config(self.CONFIG_DIR / "model_config.yaml")
        self.mcp_config = load_yaml_config(self.CONFIG_DIR / "mcp_config.yaml")
        self.cache_config = load_yaml_config(self.CONFIG_DIR / "cache_config.yaml")
        self.prompts = load_yaml_config(self.CONFIG_DIR / "prompts.yaml")
        self.logging_config = load_yaml_config(self.CONFIG_DIR / "logging_config.yaml")

    def _ensure_dirs(self):
        """Ensure necessary runtime directories exist."""
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.EXPORT_DIR.mkdir(parents=True, exist_ok=True)

settings = Settings()
