async def test_ready_ok_when_dependencies_reachable(client):
    resp = await client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["checks"] == {"postgres": "ok", "redis": "ok"}


async def test_ready_returns_503_when_redis_down(degraded_client):
    resp = await degraded_client.get("/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["checks"]["redis"] == "unavailable"
    assert body["checks"]["postgres"] == "ok"
