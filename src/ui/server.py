"""Web inspection and monitoring server for Finance RAG.

Serves an interactive dashboard using aiohttp to inspect and monitor
every step of the Corrective RAG pipeline.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
import time
from aiohttp import web
from aiohttp.typedefs import Handler
from qdrant_client import QdrantClient

from config.settings import get_settings
from src.embedding.embedder import BGEEmbedder
from src.orchestration.graph import build_crag_graph, run_query
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.parent_expander import ParentExpander
from src.retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> web.Application:
    """Create and configure the aiohttp web application."""
    app = web.Application()
    settings = get_settings()

    # Lazy or startup initialization of the pipeline components
    app["settings"] = settings
    app["graph"] = None
    app["graph_lock"] = asyncio.Lock()

    async def init_pipeline() -> None:
        if app["graph"] is None:
            logger.info("Initializing CRAG pipeline components...")
            qdrant_client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )
            embedder = BGEEmbedder(
                model_name=settings.embedding_model,
                use_fp16=settings.use_fp16,
            )
            searcher = HybridSearcher(
                client=qdrant_client,
                embedder=embedder,
                collection_name=settings.qdrant_collection,
            )
            reranker = CrossEncoderReranker(
                model_name=settings.reranker_model,
            )
            expander = ParentExpander(
                client=qdrant_client,
                collection_name=settings.qdrant_collection,
            )
            app["graph"] = build_crag_graph(
                searcher=searcher,
                reranker=reranker,
                expander=expander,
                qdrant_client=qdrant_client,
                collection_name=settings.qdrant_collection,
            )
            logger.info("CRAG pipeline ready.")

    async def handle_index(request: web.Request) -> web.FileResponse:
        """Serve the main UI page."""
        index_file = STATIC_DIR / "index.html"
        return web.FileResponse(index_file)

    async def handle_health(request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({
            "status": "online",
            "model": settings.openai_model,
            "qdrant_url": settings.qdrant_url,
            "collection": settings.qdrant_collection,
            "pipeline_ready": app["graph"] is not None,
        })

    async def handle_sample_queries(request: web.Request) -> web.Response:
        """Return curated sample queries from custom_eval.jsonl."""
        eval_path = Path(__file__).parents[2] / "data" / "eval" / "custom_eval.jsonl"
        samples = []
        if eval_path.exists():
            with open(eval_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        try:
                            item = json.loads(line)
                            samples.append({
                                "question": item.get("question"),
                                "category": item.get("category", "extraction"),
                                "company": item.get("company", "General"),
                                "difficulty": item.get("difficulty", "easy"),
                            })
                        except Exception:
                            continue
        return web.json_response({"samples": samples})

    async def handle_reload(request: web.Request) -> web.Response:
        """Reload pipeline graph and modules in memory."""
        import importlib
        import src.orchestration.graph
        import src.orchestration.nodes.generator
        import src.orchestration.nodes.grader
        import src.orchestration.nodes.hallucination_guard
        import src.orchestration.nodes.query_decomposer
        import src.orchestration.nodes.query_rewriter
        import src.orchestration.nodes.reranker_node
        import src.orchestration.nodes.retriever

        for mod in [
            src.orchestration.nodes.retriever,
            src.orchestration.nodes.reranker_node,
            src.orchestration.nodes.grader,
            src.orchestration.nodes.generator,
            src.orchestration.nodes.hallucination_guard,
            src.orchestration.nodes.query_decomposer,
            src.orchestration.nodes.query_rewriter,
            src.orchestration.graph,
        ]:
            try:
                importlib.reload(mod)
            except Exception as e:
                logger.warning(f"Failed to reload module {mod}: {e}")

        async with app["graph_lock"]:
            app["graph"] = None
            await init_pipeline()

        return web.json_response({"status": "reloaded", "pipeline_ready": True})

    async def handle_query(request: web.Request) -> web.Response:
        """Execute a query through the CRAG pipeline and return detailed state."""
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)

        query = (data.get("query") or "").strip()
        if not query:
            return web.json_response({"error": "Query cannot be empty"}, status=400)

        async with app["graph_lock"]:
            if app["graph"] is None:
                await init_pipeline()

        graph = app["graph"]

        start_time = time.perf_counter()
        try:
            # Run query in thread pool to prevent blocking the async loop
            result = await asyncio.to_thread(run_query, graph, query)
            elapsed = time.perf_counter() - start_time

            final_answer = result.get("final_answer") or result.get("generation", "")

            # Format candidate results for frontend inspection fallback
            raw_search_results = [
                {
                    "chunk_id": r.get("chunk_id", ""),
                    "score": round(float(r.get("score", 0.0)), 4),
                    "metadata": r.get("metadata", {}),
                    "text": r.get("text", ""),
                }
                for r in result.get("search_results", [])
            ]
            raw_reranked_results = [
                {
                    "chunk_id": r.get("chunk_id", ""),
                    "score": round(float(r.get("score", 0.0)), 4),
                    "metadata": r.get("metadata", {}),
                    "text": r.get("child_text", r.get("text", "")),
                    "parent_text": r.get("parent_text", ""),
                }
                for r in result.get("reranked_results", [])
            ]

            payload = {
                "query": query,
                "final_answer": final_answer,
                "confidence": result.get("confidence", "medium"),
                "cycle_count": result.get("cycle_count", 0),
                "cost_accumulated": result.get("cost_accumulated", 0.0),
                "citations": result.get("citations", []),
                "structured_metrics": result.get("structured_metrics", []),
                "hallucination_check": result.get("hallucination_check", "pass"),
                "pipeline_trace": result.get("pipeline_trace", []),
                "search_results": raw_search_results,
                "reranked_results": raw_reranked_results,
                "latency_seconds": round(elapsed, 3),
            }
            return web.json_response(payload)

        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.exception("Pipeline execution failed")
            return web.json_response({
                "error": str(e),
                "query": query,
                "latency_seconds": round(elapsed, 3),
            }, status=500)

    @web.middleware
    async def no_cache_middleware(request: web.Request, handler: Handler) -> web.StreamResponse:
        response = await handler(request)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    app.middlewares.append(no_cache_middleware)

    # Routes
    app.router.add_get("/", handle_index)
    app.router.add_get("/api/health", handle_health)
    app.router.add_get("/api/sample-queries", handle_sample_queries)
    app.router.add_post("/api/reload", handle_reload)
    app.router.add_post("/api/query", handle_query)
    app.router.add_static("/static", STATIC_DIR)

    return app


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the web server."""
    app = create_app()
    web.run_app(app, host=host, port=port)
