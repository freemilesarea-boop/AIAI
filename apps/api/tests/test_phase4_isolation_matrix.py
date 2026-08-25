"""Phase 4 verification: the mandated A/B isolation matrix, run end to end.

Not a replacement for `test_ownership_enforcement.py` — that suite is the
adversarial one. This walks the exact matrix the phase asked for, in
order, and reports each library's contents so the result can be read
rather than inferred.
"""

from __future__ import annotations

import pytest

PAYLOAD = {
    "title": "Song",
    "prompt": "bright synth pop",
    "lyrics": "",
    "vocal_gender": "instrumental",
    "duration": 30,
    "language": "en",
    "instrumental": True,
}


async def make(http, title: str) -> str:
    r = await http.post("/v1/generations", json={**PAYLOAD, "title": title})
    assert r.status_code == 202, r.text
    return str(r.json()["generation_id"])


async def titles(http) -> list[str]:
    r = await http.get("/v1/generations?limit=50")
    assert r.status_code == 200
    return sorted(item["title"] for item in r.json()["items"])


@pytest.mark.asyncio
async def test_phase4_matrix(client, client_b, capsys):
    a1 = await make(client, "A_GENERATION_1")
    a2 = await make(client, "A_GENERATION_2")
    b1 = await make(client_b, "B_GENERATION_1")
    b2 = await make(client_b, "B_GENERATION_2")

    report: list[str] = []

    # LIBRARY
    a_lib, b_lib = await titles(client), await titles(client_b)
    report.append(f"USER A LIBRARY: {a_lib}")
    report.append(f"USER B LIBRARY: {b_lib}")
    assert a_lib == ["A_GENERATION_1", "A_GENERATION_2"]
    assert b_lib == ["B_GENERATION_1", "B_GENERATION_2"]

    # PAGINATION / COUNT
    total_a = (await client.get("/v1/generations?limit=50")).json().get("total")
    total_b = (await client_b.get("/v1/generations?limit=50")).json().get("total")
    report.append(f"TOTALS: A={total_a} B={total_b}")
    assert total_a == 2 and total_b == 2

    def record(label: str, status: int, expected: int) -> None:
        ok = "PASS" if status == expected else f"FAIL(got {status})"
        report.append(f"{label}: {ok}")
        assert status == expected, f"{label}: expected {expected}, got {status}"

    # GET
    record("A gets A1", (await client.get(f"/v1/generations/{a1}")).status_code, 200)
    record("A gets B1", (await client.get(f"/v1/generations/{b1}")).status_code, 404)
    record("B gets B1", (await client_b.get(f"/v1/generations/{b1}")).status_code, 200)
    record("B gets A1", (await client_b.get(f"/v1/generations/{a1}")).status_code, 404)

    # AUDIO / DOWNLOAD
    record("A audio B1", (await client.get(f"/v1/generations/{b1}/audio")).status_code, 404)
    record(
        "A download B1",
        (await client.get(f"/v1/generations/{b1}/audio?download=true")).status_code,
        404,
    )

    # EDIT
    record(
        "A edits A1",
        (await client.patch(f"/v1/generations/{a1}", json={"title": "renamed"})).status_code,
        200,
    )
    record(
        "A edits B1",
        (await client.patch(f"/v1/generations/{b1}", json={"title": "hijack"})).status_code,
        404,
    )

    # DELETE
    record("A deletes A2", (await client.delete(f"/v1/generations/{a2}")).status_code, 204)
    record("A deletes B2", (await client.delete(f"/v1/generations/{b2}")).status_code, 404)
    assert "B_GENERATION_2" in await titles(client_b), "B2 must survive A's delete attempt"
    report.append("B2 survived A's delete: PASS")

    # LINEAGE / QA
    record("A lineage B1", (await client.get(f"/v1/generations/{b1}/lineage")).status_code, 404)
    record("A QA of B1", (await client.get(f"/v1/generations/{b1}/qa")).status_code, 404)

    # PROJECTS
    pa = (await client.post("/v1/projects", json={"name": "A album"})).json()["id"]
    pb = (await client_b.post("/v1/projects", json={"name": "B album"})).json()["id"]
    record("A opens A project", (await client.get(f"/v1/projects/{pa}")).status_code, 200)
    record("A opens B project", (await client.get(f"/v1/projects/{pb}")).status_code, 404)
    a_projects = [p["name"] for p in (await client.get("/v1/projects")).json()["items"]]
    report.append(f"A PROJECTS: {a_projects}")
    assert a_projects == ["A album"]

    # BULK: a mixed batch must not touch the foreign row.
    mixed = await client.post("/v1/generations/bulk-delete", json={"ids": [a1, b1]})
    report.append(f"BULK mixed [A1,B1] -> {mixed.status_code}")
    assert "B_GENERATION_1" in await titles(client_b), "B1 must survive a mixed bulk delete"
    report.append("B1 survived mixed bulk delete: PASS")

    # NO INFORMATION LEAK: a foreign 404 body must not carry B's data.
    body = (await client.get(f"/v1/generations/{b1}")).text
    for secret in ("B_GENERATION_1", "B album", "storage", "bright synth pop"):
        assert secret not in body, f"foreign 404 leaked {secret!r}"
    report.append("FOREIGN 404 LEAK: none")

    with capsys.disabled():
        print("\n\n=== PHASE 4 MATRIX ===")
        for line in report:
            print("  " + line)
        print()
