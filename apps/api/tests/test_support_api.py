"""Support inquiries at the HTTP boundary.

The ownership tests are the ones that matter. A support ticket holds
whatever a frustrated customer typed — an order reference, a phone
number, a description of what went wrong with their payment — and it is
exactly the sort of record an enumeration attack goes looking for.

So the question these ask is not "does the UI hide other people's
tickets" but "can the server be made to return one at all".
"""

from __future__ import annotations

from httpx import AsyncClient

PAYLOAD = {
    "category": "BILLING",
    "subject": "결제가 두 번 청구된 것 같습니다",
    "message": "8월 28일에 Basic 결제가 두 번 표시됩니다. 확인 부탁드립니다.",
}


async def _file(client: AsyncClient, **overrides: object) -> dict:
    response = await client.post("/v1/support/inquiries", json={**PAYLOAD, **overrides})
    assert response.status_code == 201, response.text
    return response.json()


# ── creating ─────────────────────────────────────────────────────────


async def test_filing_requires_a_session(anon_client: AsyncClient) -> None:
    assert (await anon_client.post("/v1/support/inquiries", json=PAYLOAD)).status_code == 401


async def test_a_valid_inquiry_is_filed(client: AsyncClient) -> None:
    body = await _file(client)

    assert body["subject"] == PAYLOAD["subject"]
    assert body["category"] == "BILLING"
    assert body["status"] == "OPEN"
    assert body["reference"].startswith("SUP-")


async def test_the_reference_is_returned_and_the_database_id_is_not(
    client: AsyncClient,
) -> None:
    """The UUID is a database key and stays on the server."""
    body = await _file(client)

    assert "id" not in body
    assert "user_id" not in body


async def test_two_inquiries_get_different_references(client: AsyncClient) -> None:
    first = await _file(client)
    second = await _file(client, subject="다른 문의")

    assert first["reference"] != second["reference"]


async def test_an_unknown_category_is_refused(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/support/inquiries", json={**PAYLOAD, "category": "REFUND_EVERYTHING"}
    )

    assert response.status_code == 422


async def test_an_empty_subject_is_refused(client: AsyncClient) -> None:
    assert (
        await client.post("/v1/support/inquiries", json={**PAYLOAD, "subject": ""})
    ).status_code == 422


async def test_a_whitespace_only_subject_is_refused(client: AsyncClient) -> None:
    """`min_length` counts characters, which is not the same question."""
    response = await client.post("/v1/support/inquiries", json={**PAYLOAD, "subject": "    "})

    assert response.status_code == 422


async def test_an_empty_message_is_refused(client: AsyncClient) -> None:
    assert (
        await client.post("/v1/support/inquiries", json={**PAYLOAD, "message": ""})
    ).status_code == 422


async def test_an_oversized_subject_is_refused(client: AsyncClient) -> None:
    response = await client.post("/v1/support/inquiries", json={**PAYLOAD, "subject": "가" * 500})

    assert response.status_code == 422


async def test_an_oversized_message_is_refused(client: AsyncClient) -> None:
    """A bound on the payload, so one request cannot be a denial of
    service against the operator queue or the database."""
    response = await client.post(
        "/v1/support/inquiries", json={**PAYLOAD, "message": "가" * 20_000}
    )

    assert response.status_code == 422


async def test_the_subject_is_stored_trimmed(client: AsyncClient) -> None:
    body = await _file(client, subject="  공백이 있는 제목  ")

    assert body["subject"] == "공백이 있는 제목"


# ── what a client may not set ────────────────────────────────────────


async def test_a_client_supplied_user_id_is_refused(
    client: AsyncClient, client_b: AsyncClient
) -> None:
    """The schema forbids unknown fields, so an attempt to file on
    somebody else's behalf does not reach the repository at all."""
    response = await client.post(
        "/v1/support/inquiries",
        json={**PAYLOAD, "user_id": client_b.user_id},  # type: ignore[attr-defined]
    )

    assert response.status_code == 422
    assert (await client_b.get("/v1/support/inquiries")).json()["total"] == 0


async def test_a_client_supplied_status_is_refused(client: AsyncClient) -> None:
    """Status is operator-owned. A customer cannot open a ticket already
    marked resolved."""
    response = await client.post("/v1/support/inquiries", json={**PAYLOAD, "status": "RESOLVED"})

    assert response.status_code == 422


async def test_a_client_supplied_reference_is_refused(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/support/inquiries", json={**PAYLOAD, "reference": "SUP-CHOSEN1"}
    )

    assert response.status_code == 422


async def test_everything_files_as_open(client: AsyncClient) -> None:
    for category in ("BILLING", "GENERATION", "DOWNLOAD", "ACCOUNT", "BUG", "FEATURE", "OTHER"):
        body = await _file(client, category=category, subject=f"{category} 문의")
        assert body["status"] == "OPEN"


# ── reading your own ─────────────────────────────────────────────────


async def test_listing_requires_a_session(anon_client: AsyncClient) -> None:
    assert (await anon_client.get("/v1/support/inquiries")).status_code == 401


async def test_an_account_with_no_inquiries_gets_an_empty_list(client: AsyncClient) -> None:
    body = (await client.get("/v1/support/inquiries")).json()

    assert body["items"] == []
    assert body["total"] == 0


async def test_the_newest_inquiry_is_first(client: AsyncClient) -> None:
    await _file(client, subject="첫 번째")
    await _file(client, subject="두 번째")

    items = (await client.get("/v1/support/inquiries")).json()["items"]

    assert [i["subject"] for i in items] == ["두 번째", "첫 번째"]


async def test_the_list_omits_the_message_body(client: AsyncClient) -> None:
    """The list is a list. The body belongs to the detail view."""
    await _file(client)

    item = (await client.get("/v1/support/inquiries")).json()["items"][0]

    assert "message" not in item


async def test_an_inquiry_can_be_opened_by_its_reference(client: AsyncClient) -> None:
    filed = await _file(client)

    body = (await client.get(f"/v1/support/inquiries/{filed['reference']}")).json()

    assert body["message"] == PAYLOAD["message"]
    assert body["reference"] == filed["reference"]


async def test_an_unknown_reference_is_404(client: AsyncClient) -> None:
    assert (await client.get("/v1/support/inquiries/SUP-NOTREAL")).status_code == 404


async def test_pagination_bounds_are_enforced(client: AsyncClient) -> None:
    assert (await client.get("/v1/support/inquiries?limit=0")).status_code == 422
    assert (await client.get("/v1/support/inquiries?limit=1000")).status_code == 422
    assert (await client.get("/v1/support/inquiries?offset=-1")).status_code == 422


# ── isolation and IDOR ───────────────────────────────────────────────


async def test_one_account_cannot_list_anothers_inquiries(
    client: AsyncClient, client_b: AsyncClient
) -> None:
    await _file(client)

    body = (await client_b.get("/v1/support/inquiries")).json()

    assert body["items"] == []
    assert body["total"] == 0


async def test_a_valid_foreign_reference_answers_404(
    client: AsyncClient, client_b: AsyncClient
) -> None:
    """The IDOR test.

    B holds a reference that really exists. The lookup puts the owner in
    the same WHERE, so it does not resolve — and the answer is the same
    404 an unknown reference gets, because "that ticket is not yours"
    would confirm it is real.
    """
    filed = await _file(client)

    response = await client_b.get(f"/v1/support/inquiries/{filed['reference']}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Inquiry not found."


async def test_the_foreign_and_unknown_answers_are_indistinguishable(
    client: AsyncClient, client_b: AsyncClient
) -> None:
    filed = await _file(client)

    foreign = await client_b.get(f"/v1/support/inquiries/{filed['reference']}")
    unknown = await client_b.get("/v1/support/inquiries/SUP-NOTREAL")

    assert foreign.status_code == unknown.status_code
    assert foreign.json() == unknown.json()


async def test_each_account_sees_only_its_own(client: AsyncClient, client_b: AsyncClient) -> None:
    await _file(client, subject="A의 문의")
    await _file(client_b, subject="B의 문의")

    a = (await client.get("/v1/support/inquiries")).json()
    b = (await client_b.get("/v1/support/inquiries")).json()

    assert [i["subject"] for i in a["items"]] == ["A의 문의"]
    assert [i["subject"] for i in b["items"]] == ["B의 문의"]


async def test_there_is_no_route_that_edits_or_deletes_an_inquiry(app) -> None:
    """A load-bearing absence.

    Support history is a record of what was said and when. Letting a
    customer rewrite or remove it after the fact would make it useless
    for resolving the disagreement it exists to document.
    """

    def walk(routes: object) -> list[tuple[str, str]]:
        found: list[tuple[str, str]] = []
        for route in routes:  # type: ignore[attr-defined]
            path = getattr(route, "path", None)
            for method in getattr(route, "methods", set()) or set():
                if path:
                    found.append((method, path))
            inner = getattr(route, "original_router", None)
            nested = getattr(inner, "routes", None) or getattr(route, "routes", None)
            if nested:
                found.extend(walk(nested))
        return found

    seen = walk(app.routes)
    assert ("POST", "/v1/support/inquiries") in seen

    # Scoped to the customer-facing prefix. The operator console added
    # `PATCH /v1/admin/support/{reference}`, which is a different
    # surface: it is behind `require_admin`, it can only move a status or
    # attach an internal note, and `TicketUpdateRequest` forbids extra
    # fields so it cannot reach the subject or the message at all — see
    # `test_an_operator_cannot_rewrite_what_a_customer_wrote`. What this
    # test protects is that the *customer* cannot rewrite their own
    # history, and that is unchanged.
    mutating = [
        (m, p) for m, p in seen if p.startswith("/v1/support") and m in {"PUT", "PATCH", "DELETE"}
    ]
    assert mutating == []


# ── content safety ───────────────────────────────────────────────────


async def test_markup_in_a_message_is_stored_verbatim(client: AsyncClient) -> None:
    """A ticket saying `<script>` is a customer describing a bug.

    It is stored as typed and returned as typed; nothing renders it as
    markup, and the browser escapes it. Mangling it here would corrupt a
    legitimate bug report.
    """
    payload = "<script>alert('x')</script> 이 문자열이 화면에 그대로 보입니다"
    filed = await _file(client, message=payload)

    body = (await client.get(f"/v1/support/inquiries/{filed['reference']}")).json()

    assert body["message"] == payload


async def test_a_context_url_is_optional(client: AsyncClient) -> None:
    body = await _file(client)

    assert body["context_url"] is None


async def test_a_context_url_is_kept_when_given(client: AsyncClient) -> None:
    body = await _file(client, context_url="https://boorda.kr/library")

    assert body["context_url"] == "https://boorda.kr/library"


# ── account closure compatibility ────────────────────────────────────


async def test_an_account_with_inquiries_can_still_be_closed(client: AsyncClient) -> None:
    """Support history must not become a reason account closure fails.

    `support_tickets.user_id` restricts rather than cascades, matching
    `generations` — and closing an account is anonymisation, so the
    ticket stays attached to a row that no longer names anyone.
    """
    await _file(client)

    response = await client.post(
        "/v1/auth/account/delete", json={"current_password": "correct horse battery staple"}
    )

    assert response.status_code == 204


async def test_a_closed_account_cannot_read_its_inquiries(client: AsyncClient) -> None:
    await _file(client)
    await client.post(
        "/v1/auth/account/delete", json={"current_password": "correct horse battery staple"}
    )

    assert (await client.get("/v1/support/inquiries")).status_code == 401


async def test_the_inquiry_survives_the_account_closing(app, client: AsyncClient) -> None:
    """The complaint outlives the complainant leaving, which is when it
    matters most."""
    from sqlalchemy import func, select

    from luber_database.models.support import SupportTicket

    await _file(client)
    await client.post(
        "/v1/auth/account/delete", json={"current_password": "correct horse battery staple"}
    )

    factory = app.state.session_factory
    async with factory() as session:
        remaining = (
            await session.execute(select(func.count()).select_from(SupportTicket))
        ).scalar_one()

    assert remaining == 1
