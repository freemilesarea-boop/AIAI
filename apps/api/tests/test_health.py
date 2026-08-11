async def test_health_returns_ok(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"status": "ok", "service": "luber-api"}


async def test_health_sets_request_id_header(client):
    resp = await client.get("/health")
    assert resp.headers.get("x-request-id")


async def test_request_id_is_propagated_when_supplied(client):
    resp = await client.get("/health", headers={"X-Request-ID": "req-abc-123"})
    assert resp.headers["x-request-id"] == "req-abc-123"
