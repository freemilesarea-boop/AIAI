"""The outbound half: registering and cancelling recurring payments.

Two calls, both to PayApp's single REST endpoint
``https://api.payapp.kr/oapi/apiLoad.html`` as
``application/x-www-form-urlencoded`` UTF-8, and both returning a
URL-encoded ``key=value`` body rather than JSON.

The client is defined as a Protocol first and implemented second, so
every test in this repository runs against a deterministic fake and no
automated run can ever reach PayApp. That is not a testing convenience:
a suite that could accidentally call ``rebillRegist`` against real
credentials is a suite that can register a real recurring contract
against a real customer.

**What a successful registration means.** ``state=1`` from
``rebillRegist`` means PayApp accepted the *request*. No card has been
charged and the customer has not even opened the payment window yet.
Nothing in this module returns anything that could be mistaken for
payment, which is why the result type is called
:class:`RebillRegistration` and carries no notion of success beyond
"registered".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import parse_qsl

import httpx

logger = logging.getLogger(__name__)

#: PayApp's only API endpoint. Every command is a `cmd` field on a POST.
DEFAULT_API_URL = "https://api.payapp.kr/oapi/apiLoad.html"

#: PayApp documents 30 seconds as the recommended timeout. A checkout
#: that hangs longer than this is one the user has already given up on,
#: and the local record written *before* the call means a timeout is
#: recoverable rather than a gap.
DEFAULT_TIMEOUT_SECONDS = 30.0

#: `rebillCycleType=Month` with a day-of-month. BOORDA's billing policy
#: is documented in `policy.py`; this module only transmits it.
CYCLE_TYPE_MONTH = "Month"


class PayAppError(Exception):
    """The provider refused the request or could not be reached.

    ``errno``/``message`` come from PayApp when it answered. Both may be
    absent when the failure was transport-level — a timeout has no error
    code, and inventing one would make a network problem look like a
    business rejection in the logs.
    """

    def __init__(self, message: str, *, errno: str | None = None, transport: bool = False) -> None:
        super().__init__(message)
        self.errno = errno
        #: True when we never got an answer. The caller must treat this
        #: as *unknown*, not as failure: PayApp may have registered the
        #: contract and lost the response.
        self.transport = transport


@dataclass(frozen=True)
class RebillRegistration:
    """PayApp accepted a recurring registration. Nobody has paid yet."""

    rebill_no: str
    payurl: str


class PayAppClient(Protocol):
    """What the billing service needs from a payment provider."""

    async def register_recurring(
        self,
        *,
        goodname: str,
        goodprice: int,
        recvphone: str,
        cycle_day: int,
        expire_date: str,
        feedbackurl: str,
        failurl: str,
        returnurl: str,
        var1: str,
        var2: str | None = None,
    ) -> RebillRegistration: ...

    async def cancel_recurring(self, *, rebill_no: str) -> None: ...


def _parse_response(body: str) -> dict[str, str]:
    """PayApp answers with `key=value&key=value`, not JSON."""
    return dict(parse_qsl(body.strip(), keep_blank_values=True))


class HttpPayAppClient:
    """The real client. Constructed only where credentials are configured."""

    def __init__(
        self,
        *,
        userid: str,
        linkkey: str,
        api_url: str = DEFAULT_API_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._userid = userid
        self._linkkey = linkkey
        self._api_url = api_url
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _call(self, fields: dict[str, Any]) -> dict[str, str]:
        payload = {"userid": self._userid, "linkkey": self._linkkey, **fields}
        try:
            response = await self._client.post(
                self._api_url,
                data=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded; charset=utf-8"},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            # No `payload` in this log line, ever: it holds linkkey.
            logger.warning(
                "payapp request failed at transport level",
                extra={"payapp_cmd": fields.get("cmd"), "error_type": type(exc).__name__},
            )
            raise PayAppError(str(exc), transport=True) from exc

        parsed = _parse_response(response.text)
        if parsed.get("state") != "1":
            errno = parsed.get("errno")
            # PayApp's message can be shown to an operator but never to
            # the customer unmodified — it is provider-shaped text about
            # our merchant account.
            message = parsed.get("errorMessage") or "payapp rejected the request"
            logger.warning(
                "payapp rejected request",
                extra={"payapp_cmd": fields.get("cmd"), "payapp_errno": errno},
            )
            raise PayAppError(message, errno=errno)
        return parsed

    async def register_recurring(
        self,
        *,
        goodname: str,
        goodprice: int,
        recvphone: str,
        cycle_day: int,
        expire_date: str,
        feedbackurl: str,
        failurl: str,
        returnurl: str,
        var1: str,
        var2: str | None = None,
    ) -> RebillRegistration:
        fields: dict[str, Any] = {
            "cmd": "rebillRegist",
            "goodname": goodname,
            "goodprice": goodprice,
            "recvphone": recvphone,
            "rebillCycleType": CYCLE_TYPE_MONTH,
            "rebillCycleMonth": cycle_day,
            "rebillExpire": expire_date,
            "feedbackurl": feedbackurl,
            "failurl": failurl,
            "returnurl": returnurl,
            "var1": var1,
            # PayApp retries the feedback URL when the response is not
            # exactly `SUCCESS`, up to ten times. That retry is what lets
            # the endpoint refuse to acknowledge an event it could not
            # durably record, instead of having to choose between lying
            # and losing the payment.
            "checkretry": "y",
        }
        if var2 is not None:
            fields["var2"] = var2

        parsed = await self._call(fields)
        rebill_no = parsed.get("rebill_no")
        payurl = parsed.get("payurl")
        if not rebill_no or not payurl:
            # state=1 without the identifiers is not something the
            # documentation describes. Treated as a failure rather than
            # guessed at: a subscription with no rebill_no could never be
            # cancelled or reconciled.
            raise PayAppError("payapp accepted the registration but returned no rebill_no/payurl")
        return RebillRegistration(rebill_no=rebill_no, payurl=payurl)

    async def cancel_recurring(self, *, rebill_no: str) -> None:
        """Stop the next charge. Does not refund the last one.

        PayApp: 「다음 결제 주기에 정기 결제가 발생되지 않습니다」 — future
        charges stop; an approved payment cannot be reverted this way.
        """
        await self._call({"cmd": "rebillCancel", "rebill_no": rebill_no})


__all__ = [
    "CYCLE_TYPE_MONTH",
    "DEFAULT_API_URL",
    "DEFAULT_TIMEOUT_SECONDS",
    "HttpPayAppClient",
    "PayAppClient",
    "PayAppError",
    "RebillRegistration",
]
