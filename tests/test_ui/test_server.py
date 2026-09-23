"""Unit tests for the Finance RAG Web UI server endpoints."""

import pytest
from aiohttp.test_utils import TestClient, TestServer
from src.ui.server import create_app


@pytest.mark.asyncio
async def test_health_endpoint():
    app = create_app()
    client = TestClient(TestServer(app))
    await client.start_server()

    try:
        resp = await client.get("/api/health")
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "online"
        assert "model" in data
        assert "qdrant_url" in data
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_sample_queries_endpoint():
    app = create_app()
    client = TestClient(TestServer(app))
    await client.start_server()

    try:
        resp = await client.get("/api/sample-queries")
        assert resp.status == 200
        data = await resp.json()
        assert "samples" in data
        assert isinstance(data["samples"], list)
        if len(data["samples"]) > 0:
            assert "question" in data["samples"][0]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_static_index_endpoint():
    app = create_app()
    client = TestClient(TestServer(app))
    await client.start_server()

    try:
        resp = await client.get("/")
        assert resp.status == 200
        text = await resp.text()
        assert "Finance RAG" in text
        assert "CRAG Pipeline Inspector" in text
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_query_empty_validation():
    app = create_app()
    client = TestClient(TestServer(app))
    await client.start_server()

    try:
        resp = await client.post("/api/query", json={"query": ""})
        assert resp.status == 400
        data = await resp.json()
        assert "error" in data
    finally:
        await client.close()
