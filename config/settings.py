"""Centralized application configuration loaded from environment variables.

Uses Pydantic Settings to validate and type-check all configuration values.
Settings are loaded from a `.env` file at the project root.
"""

import os
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Load .env into os.environ so third-party SDKs (OpenAI, LangChain, etc.) can read them
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # === OpenAI ===
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o", description="OpenAI model for generation/grading")

    # === Qdrant ===
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant server URL")
    qdrant_api_key: str | None = Field(default=None, description="Qdrant API key (Cloud only)")
    qdrant_collection: str = Field(default="sec_filings", description="Qdrant collection name")

    # === Embedding ===
    embedding_model: str = Field(default="BAAI/bge-m3", description="BGE-M3 embedding model")
    embedding_dim: int = Field(default=1024, description="Dense embedding dimension")
    use_fp16: bool = Field(default=True, description="Use FP16 for embedding model")

    # === Reranker ===
    reranker_model: str = Field(
        default="BAAI/bge-reranker-large", description="Cross-encoder reranker model"
    )

    # === Retrieval ===
    retrieval_top_k: int = Field(default=25, description="Top-K candidates from hybrid search")
    rerank_top_k: int = Field(default=5, description="Top-K after reranking (default)")
    rerank_top_k_expanded: int = Field(
        default=8, description="Top-K after reranking (multi-entity/multi-year)"
    )

    # === CRAG ===
    max_crag_cycles: int = Field(default=2, description="Maximum CRAG correction cycles")
    reranker_coarse_threshold: float = Field(
        default=-2.0,
        description="Stage 1 coarse gate: top rerank score below this triggers immediate abstention",
    )

    # === Chunking ===
    prose_chunk_size: int = Field(default=512, description="Target prose chunk size in tokens")
    prose_chunk_overlap: float = Field(
        default=0.1, description="Overlap ratio between prose chunks"
    )
    max_table_chunk_tokens: int = Field(
        default=2048, description="Max tokens before splitting a table into children"
    )

    # === SEC EDGAR ===
    sec_user_agent_company: str = Field(
        default="FinanceRAG", description="Company name for SEC User-Agent header"
    )
    sec_user_agent_email: str = Field(
        default="", description="Contact email for SEC User-Agent header"
    )

    # === Cost Tracking ===
    budget_per_query: float = Field(
        default=0.10, description="Max USD cost per single query"
    )
    budget_per_eval_run: float = Field(
        default=50.0, description="Max USD cost per evaluation run"
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


def get_settings() -> Settings:
    """Factory function to create Settings instance. Allows overriding in tests."""
    s = Settings()
    if s.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = s.openai_api_key
    return s
