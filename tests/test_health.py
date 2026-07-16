"""Smoke test: FastAPI /health returns 200."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from biolit.api.main import create_app


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    app = create_app()

    async def _ok() -> dict[str, Any]:
        return {"ok": True}

    with (
        patch("biolit.api.main.ping_db", new=AsyncMock(side_effect=_ok)),
        patch("biolit.api.main.ping_redis", new=AsyncMock(side_effect=_ok)),
        patch("biolit.api.main.close_redis", new=AsyncMock()),
        patch("biolit.api.main.dispose_engine", new=AsyncMock()),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


@pytest.mark.asyncio
async def test_health_returns_200(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app"] == "biolit"
    assert body["db"]["ok"] is True
    assert body["redis"]["ok"] is True
